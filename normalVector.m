function [V1,V2,V3] = normalVector(pointKnn)
% Time:2021.12.13
% Reference function:
% Improvement: 
% Author:YZHLhappy
%--------------------------------------------------------------------------
% This function is used to calculate the normal vector of point cloud.
%--------------------------------------------------------------------------
% Input: pointKnn
% pointKnn: points with x,y,z coordinates. 
%           M x 3, M is the number of point.

% Output: result
% reult: normal vector

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
[V,D] = eig(C);
[~,idx] = min(diag(D));
% Normalize
V = V(:,idx)./norm(V(:,idx));
V1 = V(1);
V2 = V(2);
V3 = V(3);
end
