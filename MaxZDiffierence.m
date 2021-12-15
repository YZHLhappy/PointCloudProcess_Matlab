function [result] = MaxZDiffierence(pointKnn)
% Time:2021.12.13
% Reference function:
% Improvement: 
% Author:YZHLhappy
%--------------------------------------------------------------------------
% This function is used to calculate the max difference of Z-coordinate of 
% point cloud.
%--------------------------------------------------------------------------
% Input: pointKnn
% pointKnn: points with x,y,z coordinates. 
%           M x 3, M is the number of point.

% Output: result
% reult: the max difference of Z-coordinate of point cloud.

result = max(pointKnn(:,3)) - min(pointKnn(:,3));
end