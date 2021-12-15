function [result] = Planarity(pointKnn)
% Time:2021.12.13
% Reference function:
% Improvement: 
% Author:YZHLhappy
%--------------------------------------------------------------------------
% This function is used to calculate the Planarity of point cloud.
%--------------------------------------------------------------------------
% Input: pointKnn
% pointKnn: points with x,y,z coordinates. 
%           M x 3, M is the number of point.

% Output: result
% reult: Planarity

mx = mean(pointKnn(:,1));
my = mean(pointKnn(:,2));
mz = mean(pointKnn(:,3));

[n,~] = size(pointKnn);

for i=1:n
    C1(i,1) = pointKnn(i,1)-mx;
    C1(i,2) = pointKnn(i,2)-my;
    C1(i,3) = pointKnn(i,3)-mz;
end
C=C1'*C1;
[~,D] = eig(C);
result = (D(2,2)-D(3,3))/D(1,1);
end