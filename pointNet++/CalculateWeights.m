function weights = CalculateWeights(labels, numClasses)
% from matlab example: https://www.mathworks.com/help/lidar/ug/aerial-lidar-segmentation-using-pointnet-network.html
% helperCalculateWeights computes weights of each class in the point cloud.
%
% This is an example helper function that is subject to change or removal
% in future releases.

% Copyright 2021 MathWorks, Inc.
weights = zeros(1,numClasses);
for i=1:numClasses
    weights(i) = sum(labels==i);
end
end