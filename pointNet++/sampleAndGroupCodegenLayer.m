classdef sampleAndGroupCodegenLayer < nnet.layer.Layer
% sampleAndGroupCodegenLayer is a custom layer with code generation support
% which is used to replace the functionLayer for the sampleAndGroupFcn
% present in the PointNet++ network. This layer has to be used only in the
% code generation workflow to generate CUDA MEX code.
%
% This is an example custom layer that is subject to change or removal in
% future releases.

% Copyright 2021 MathWorks, Inc.

    %#codegen

    properties
        location
        pointsDim
        numClusters
        clusterSize
        radius
        catXYZ
    end

    methods
        function layer = sampleAndGroupCodegenLayer(name,location,pointsDim,numClusters,clusterSize,radius,layerDescription)
            layer.location = location;
            layer.pointsDim = pointsDim;
            layer.numClusters = numClusters;
            layer.clusterSize = clusterSize;
            layer.radius = radius;
            layer.catXYZ = true;

            if strcmp(location,'Initial')
                InputNames = {'points'};
                OutputNames = {'newPointFeatures','centroids'};
            else
                InputNames = {'pointFeatures','points'};
                OutputNames = {'newPointFeatures','centroids'};
            end

            layer.InputNames = InputNames;
            layer.OutputNames = OutputNames;
            layer.Name = name;
            layer.Description = layerDescription;
        end

        function [newpointFeatures,centroids] = predict(layer,varargin)

            [newpointFeatures,centroids] = layer.sampleAndGroupFcn(layer.location,layer.pointsDim,...
                layer.numClusters,layer.clusterSize,layer.radius,layer.catXYZ,varargin{:});
        end

        function [newpointFeatures,centroids] = sampleAndGroupFcn(layer,location,pointsDim,numClusters,clusterSize,radius,catXYZ,varargin)
            % The sampleAndGroupFcn first samples the point cloud to a given number of
            % clusters and then constructs local region sets by finding neighboring
            % points around the centroids using the queryBallPoint function.
            % location        - Location of the layer in the network.
            % pointsDim       - Dimension of the input points.
            % numClusters     - Number of clusters to form.
            % clusterSize     - Number of points to group w.r.t each centroid
            % radius          - Grouping radius.
            % catXYZ          - If set to true, concatenate xyz coordinates of the
            %                   points with local features.
            % points          - xyz coordinates of the points from previous set
            %                   abstraction module.
            % pointFeatures   - pointFeatures from the previous set abstraction module.

            if strcmp(location,'Initial')
                points = extractdata(squeeze(varargin{1}));

                % If the dimension of input points is more than 3 ie., they have more
                % than xyz coordinates, then crop them and only use xyz coordinates for
                % sampling and finding the grouping indices.
                if pointsDim > 3
                    pointFeatures = points(:,4:end,:);
                    points = points(:,1:3,:);
                else
                    pointFeatures = [];
                end

            else
                points = extractdata(squeeze(varargin{2}));
                pointFeatures = extractdata(squeeze(varargin{1}));
            end

            batchSize = size(points,3);

            % Find the new cluster centers using farthest point sampling .
            centroidIdx = layer.farthestPointSampling(points,numClusters,batchSize);
            centroids = coder.nullcopy(zeros(numClusters,3,batchSize,'like',points));
            for n = 1:batchSize
                centroids(:,:,n) = points(centroidIdx(:,n),:,n);
            end

            % Find the nearest clusterSize number of samples to group the points w.r.t
            % new cluster centers.
            idx = layer.queryBallPoint(points,centroids,numClusters,clusterSize,radius,batchSize);
            newClusters = coder.nullcopy(zeros(numClusters,clusterSize,3,batchSize,'like',points));
            for n = 1:batchSize
                newClusters(:,:,:,n) = reshape(points(idx(:,:,n),:,n),[numClusters,clusterSize,3]);
            end

            % bsxfun for codegen
            newClusters = bsxfun(@minus,newClusters,permute(centroids,[1 4 2 3]));

            if ~isempty(pointFeatures)
                featureSize = size(pointFeatures,2);
                localFeatures = coder.nullcopy(zeros(numClusters,clusterSize,featureSize,batchSize,'like',points));
                for n = 1:batchSize
                    localFeatures(:,:,:,n) = reshape(pointFeatures(idx(:,:,n),:,n),[numClusters,clusterSize,featureSize,1]);
                end

                if catXYZ
                    newpointFeatures = cat(3,newClusters,localFeatures);
                else
                    newpointFeatures = localFeatures;
                end
            else
                newpointFeatures = newClusters;
            end

            % To maintain consistent batch size for codegen
            centroids = permute(centroids,[1 2 4 3]);
            centroids = dlarray(centroids);
            newpointFeatures = dlarray(newpointFeatures);
        end

        function centroidIdx = farthestPointSampling(~,points,numClusters,batchSize)
            % The farthestPointSampling function selects a set of points from input
            % points which defines the new cluster centers.
            % points - PointCloud locations Nx3.
            % numClusters - Number of clusters to find.
            % centroidIdx - Indices of the new cluster centers.

            % Initialize initial indices as zeros.
            centroidIdx = coder.nullcopy(zeros(numClusters,batchSize,'like',points));

            % Distance from centroid to each point.
            distance = ones(size(points,1),1,batchSize,'like',points) .* 1e10;

            % Random initialization of the first point.
            farthest = randi([1,size(points,1)],1,batchSize,'like',points);

            for i = 1:numClusters
                centroidIdx(i,:) = farthest;
                centroid = coder.nullcopy(zeros(1,3,batchSize,'like',points));
                for k = 1:batchSize
                    centroid(:,:,k) = points(farthest(k),:,k);
                end
                % bsxfun for codegen
                dist = sum(((bsxfun(@minus, points, centroid)).^2),2);
                mask = dist < distance;
                distance(mask) = dist(mask);
                [~,farthestTemp] = max(distance,[],1);
                farthest = cast(farthestTemp,'like',farthest);
            end
        end

        function groupIdx = queryBallPoint(layer,points,centroids,numClusters,clusterSize,radius,batchSize)
            % Given a cluster center the queryBallPoint finds all points that are
            % within a radius to the query point.

            N = ones(1,1,'like',points)*size(points,1);
            groupIdx = repmat(1:N,[numClusters 1 batchSize]);

            % Find distance between centroids and given points.
            sqDist = layer.squareDistance(centroids,points);

            % Find points that are inside given radius.
            groupIdx(sqDist > (radius)^2) = N;
            groupIdx = gpucoder.sort(groupIdx,2,"ascend");

            % Find the closest clusterSize points within the given radius.
            groupIdx = groupIdx(:,1:clusterSize,:);
            groupFirst = repmat(groupIdx(:,1,:),1,clusterSize);
            mask = (groupIdx == N);
            groupIdx(mask) = groupFirst(mask);
        end

        function dist = squareDistance(~,src,dst)
            % Squared distance between two set of points.
            dist = -2 * (src*permute(dst,[2,1,3]));
            tmp1 = sum(src.^2,2);
            tmp1 = reshape(tmp1,size(src,1),1,[]);
            tmp2 = sum(dst.^2,2);
            tmp2 = reshape(tmp2,1,size(dst,1),[]);
            % bsxfun for codegen
            dist = bsxfun(@plus,bsxfun(@plus,dist,tmp1),tmp2);
        end

    end
end