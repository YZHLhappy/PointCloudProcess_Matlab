function [result] = MeanZ(pointKnn)
% Time:2021.12.13
% Reference function:
% Improvement: 
% Author:YZHLhappy
%--------------------------------------------------------------------------
% This function is used to calculate the mean value of Z-coordinate of 
% point cloud.
%--------------------------------------------------------------------------
% Input: pointKnn
% pointKnn: points with x,y,z coordinates. 
%           M x 3, M is the number of point.

% Output: result
% reult: mean value of Z-coordinate

result = mean(pointKnn(:,3));

end