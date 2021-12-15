function [result] = VarianceZ(pointKnn)
% Time:2021.12.13
% Reference function:
% Improvement: 
% Author:YZHLhappy
%--------------------------------------------------------------------------
% This function is used to calculate the Z-coordinate's variance of point 
% cloud.
%--------------------------------------------------------------------------
% Input: pointKnn
% pointKnn: points with x,y,z coordinates. 
%           M x 3, M is the number of point.

% Output: result
% result: the Z-coordinate's variance of point cloud.

mz = mean(pointKnn(:,3));

[n,~] = size(pointKnn);

for i=1:n
    C1(i,3) = pointKnn(i,3)-mz;
end
result = sum(C1(:,3))/n;
end