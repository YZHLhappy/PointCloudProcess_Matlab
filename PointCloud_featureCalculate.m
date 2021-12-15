function [result] = PointCloud_featureCalculate(xyz,structure,para)
% Time:2021.12.13 Create this function
%      2021.12.15 add roughness
% Reference function:
% Improvement: 
% Author:YZHLhappy
%--------------------------------------------------------------------------
% This function is used to calculate the features of point cloud.
%--------------------------------------------------------------------------
% Input: xyz, structure, para
% xyz: point cloud with x,y,z coordinates. M x 3, M is the number of point;
% sturcture: the structure of point cloud;
%            structure = 0: knnsearch
%            structure = 1: rangesearch
% para: the parameter of knnsearch/rangesearch;
%       if use knnserach, k = para
%       if use rangesearch, r = para

% Output: result
% result: the features of point cloud (Input:xyz); M x 16
%         1. surface_Variance;
%         2. anisotropy;
%         3. changofCurvature;
%         4. eigenentropy;
%         5. linearity;
%         6. omnivariance;
%         7. pca2;
%         8. planarity;
%         9. sphericity; 
%         10. sumofEigenvalues;
%         11. maxZDiffierence;
%         12. meanZ;
%         13. varianceZ;
%         14~16. Nx;Ny;Nz
%         17. roughness

tic
% Construct point cloud into kdtree structure
[sizePointCloud,~] = size(xyz);
mdl = KDTreeSearcher(xyz);
% Calculate the feature of each point
% 1. knn: structure==0;
if(structure == 0)
    k = para;
    parfor j=1:sizePointCloud
        Point = xyz(j,:);  % Traverse each point, here is the j-th point
        indexNN = knnsearch(mdl,Point,"K",k);
        pointKnn = mdl.X(indexNN,:);
        % features calculate
        surface_Variance(j)= surfaceVariance(pointKnn);
        anisotropy(j) = Anisotropy(pointKnn);
        changofCurvature(j) = ChangofCurvature(pointKnn);
        eigenentropy(j)=Eigenentropy(pointKnn);
        linearity(j) = Linearity(pointKnn);
        omnivariance(j) = Omnivariance(pointKnn);
        pca2(j) = PCA2(pointKnn);
        planarity(j) = Planarity(pointKnn);
        sphericity(j) = Sphericity(pointKnn);
        sumofEigenvalues(j) = SumofEigenvalues(pointKnn);

        roughness(j) = Roughness(pointKnn,Point);

        maxZDiffierence(j) = MaxZDiffierence(pointKnn);
        meanZ(j) = MeanZ(pointKnn);
        varianceZ(j) = VarianceZ(pointKnn);


        [Nx(j),Ny(j),Nz(j)] = normalVector(pointKnn);
    end
result = [surface_Variance;anisotropy;changofCurvature;eigenentropy;linearity;...
    omnivariance;pca2;planarity;sphericity;sumofEigenvalues;maxZDiffierence;...
    meanZ;varianceZ;Nx;Ny;Nz;roughness]';
end
% 2. r: structure==1;
if(structure == 1)
    r = para;
    parfor j=1:sizePointCloud
        Point = xyz(j,:);  % Traverse each point, here is the j-th point
        indexNN = rangesearch(mdl,Point,r);
        pointKnn = mdl.X(cell2mat(indexNN),:);
        % features calculate
        surface_Variance(j)= surfaceVariance(pointKnn);
        anisotropy(j) = Anisotropy(pointKnn);
        changofCurvature(j) = ChangofCurvature(pointKnn);
        eigenentropy(j)=Eigenentropy(pointKnn);
        linearity(j) = Linearity(pointKnn);
        omnivariance(j) = Omnivariance(pointKnn);
        pca2(j) = PCA2(pointKnn);
        planarity(j) = Planarity(pointKnn);
        sphericity(j) = Sphericity(pointKnn);
        sumofEigenvalues(j) = SumofEigenvalues(pointKnn);

        roughness(j) = Roughness(pointKnn,Point);

        maxZDiffierence(j) = MaxZDiffierence(pointKnn);
        meanZ(j) = MeanZ(pointKnn);
        varianceZ(j) = VarianceZ(pointKnn);

        [Nx(j),Ny(j),Nz(j)] = normalVector(pointKnn);
    end
result = [surface_Variance;anisotropy;changofCurvature;eigenentropy;linearity;...
    omnivariance;pca2;planarity;sphericity;sumofEigenvalues;maxZDiffierence;...
    meanZ;varianceZ;Nx;Ny;Nz;roughness]';
end
toc
end