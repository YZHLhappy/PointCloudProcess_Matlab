function [result] = featureCalculate(xyz,k)



[sizePointCloud,~] = size(xyz);
mdl = KDTreeSearcher(xyz);

tic
parfor j=1:sizePointCloud
    Point = xyz(j,:);
%     k = 50;
    indexNN = knnsearch(mdl,Point,"K",k);
    pointKnn = xyz(indexNN,:);

    surface_Variance(j)= surfaceVariance(pointKnn,k);
    anisotropy(j) = Anisotropy(pointKnn,k);
    changofCurvature(j) = ChangofCurvature(pointKnn,k);
    eigenentropy(j)=Eigenentropy(pointKnn,k);
    linearity(j) = Linearity(pointKnn,k);
    omnivariance(j) = Omnivariance(pointKnn,k);
    pca2(j) = PCA2(pointKnn,k);
    planarity(j) = Planarity(pointKnn,k);
    sphericity(j) = Sphericity(pointKnn,k);
    sumofEigenvalues(j) = SumofEigenvalues(pointKnn,k);

    maxZDiffierence(j) = MaxZDiffierence(pointKnn);
    meanZ(j) = MeanZ(pointKnn);
    varianceZ(j) = VarianceZ(pointKnn,k);

    [Nx(j),Ny(j),Nz(j)] = normalVector(pointKnn,k);
end
toc
result = [surface_Variance;anisotropy;changofCurvature;eigenentropy;linearity;...
    omnivariance;pca2;planarity;sphericity;sumofEigenvalues;maxZDiffierence;...
    meanZ;varianceZ;Nx;Ny;Nz]';
end