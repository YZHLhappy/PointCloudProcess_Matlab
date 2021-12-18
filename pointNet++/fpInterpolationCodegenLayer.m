classdef fpInterpolationCodegenLayer < nnet.layer.Layer
% fpInterpolationCodegenLayer is a custom layer with code generation
% support which is used to replace the functionLayer for the
% fpInterpolationFcn present in the PointNet++ network. This layer has to
% be used only in the code generation workflow to generate CUDA MEX code.
%
% This is an example custom layer that is subject to change or removal in
% future releases.

% Copyright 2021 MathWorks, Inc.

    %#codegen

    properties
        location
        pointsDim
    end

    methods
        function layer = fpInterpolationCodegenLayer(name,location,pointsDim)
            if strcmp(location,'Final')
                InputNames = {'pointFeatures2','points2','points1'};
            else
                InputNames = {'pointFeatures2','points2','points1','pointFeatures1'};
            end

            OutputNames = {'interpolatedFeatures'};
            layerDescription = "Interpolation";
            layer.InputNames = InputNames;
            layer.OutputNames = OutputNames;
            layer.Name = name;
            layer.Description = layerDescription;
            layer.location = location;
            layer.pointsDim = pointsDim;
        end

        function interpolatedFeatures = predict(layer,pointFeatures2,points2,points1,varargin)

            interpolatedFeatures = layer.fpInterpolationFcn(layer.location,layer.pointsDim,...
                pointFeatures2,points2,points1,varargin{:});
        end

        function interpolatedFeatures = fpInterpolationFcn(layer,location,pointsDim,pointFeatures2,points2,points1,varargin)
            % Use the inverse distance weighted average based on the k nearest
            % neighbors to interpolate features.
            % For an n'th feature propagation module in a network with N set
            % abstraction modules
            % points1             - points from (N-n)th set abstraction module.
            % points2             - points from (N-n+1)th set abstraction module.
            % pointFeatures1      - pointFeatures from (N-n)th set abstraction module.
            % pointFeatures2      - pointFeatures from (n-1)th feature propagation
            %                       module.
            % pointsDim           - Dimension of the input points.
            % location            - Location of the layer in the network.
            % interpolatedFeatures- Interpolated features.

            points1 = extractdata(squeeze(points1));
            points2 =  extractdata(squeeze(points2));
            pointFeatures2 = extractdata(squeeze(pointFeatures2));


            batchSize = size(points1,3);

            if strcmp(location,'Final')
                % If the input points have more than xyz coordinates then crop them
                % and only use xyz coordinates for interpolation.
                if pointsDim > 3
                    pointFeatures1 = points1(:,4:end,:);
                    points1 = points1(:,1:3,:);
                else
                    pointFeatures1 = [];
                end
            else
                pointFeatures1 = extractdata(squeeze(varargin{1}));
            end

            % Find the K (=3) nearest neighbors for each point.
            dists = layer.squareDistance(points1,points2);
            [dists,idx] = gpucoder.sort(dists,2,"ascend");
            dists = dists(:,1:3,:);
            idx = idx(:,1:3,:);

            % Calculate the weights for interpolation.
            distRecip = 1./(dists+1e-8);
            normFactor = sum(distRecip,2);
            % bsxfun for codegen
            weights = bsxfun(@rdivide,distRecip,normFactor);

            % Perform weighted interpolation.
            szIdx = size(idx);
            interpolatedFeatures = coder.nullcopy(zeros(szIdx(1),szIdx(2),size(pointFeatures2,2),batchSize,'like',points1));
            for k = 1:batchSize
                interpolatedFeatures(:,:,:,k) = reshape(pointFeatures2(idx(:,:,k),:,k),szIdx(1),szIdx(2),size(pointFeatures2,2),1);
            end
            % bsxfun for codegen
            interpolatedFeatures = bsxfun(@times,interpolatedFeatures, permute(weights,[1 2 4 3]));
            interpolatedFeatures = squeeze(sum(interpolatedFeatures,2));

            if ~isempty(pointFeatures1)
                interpolatedFeatures = permute(cat(2,pointFeatures1,interpolatedFeatures),[1 4 2 3]);
            else
                interpolatedFeatures = permute(interpolatedFeatures,[1 4 2 3]);
            end
            interpolatedFeatures = dlarray(interpolatedFeatures);
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