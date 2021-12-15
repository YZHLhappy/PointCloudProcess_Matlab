function [roughness] = Roughness(pointKnn,Point)
% Time:2021.12.15
% Reference function:
% Improvement: 
% Author:YZHLhappy
%--------------------------------------------------------------------------
% This function is used to calculate the Roughness of point cloud.
%--------------------------------------------------------------------------
% Input: pointKnn,Point
% pointKnn: points (sed to construct the plane) with x,y,z coordinates. 
%           M x 3, M is the number of point.
% Point: current point (need to combine the definition of roughness to
%                       understand, the definition of roughness is in the 
%                       output interpretation)

% Output: roughness
% roughness: the 'roughness' value is equal to the distance between this 
%            point and the best fitting plane computed on its nearest 
%            neighbors.

[n,~] = size(pointKnn); % the number of the 3D points

mx = mean(pointKnn(:,1));
my = mean(pointKnn(:,2));
mz = mean(pointKnn(:,3));

X_center = [mx;my;mz]; % the center of the plan
M = 0;
for i=1:n
    B = pointKnn(i,:)' - X_center;
    M = M + B * B'; % The matirx of the secord central moment M
end
[Eigen_vektor,Eigen_wert] = eig(M);
Lambada = max(Eigen_wert);
[~,index_3] = min(Lambada);
U_3 = Eigen_vektor(:,index_3);
Omega_H = U_3;
Omega_O = -Omega_H'* X_center;

% homogeneousPlane: Plane in homogenous representation
homogeneousPlane = [Omega_H;Omega_O];

% Calculate the distance
on = abs(homogeneousPlane(1)*Point(1)+homogeneousPlane(2)*Point(2)+...
    homogeneousPlane(3)*Point(3)+homogeneousPlane(4));
down = sqrt(homogeneousPlane(1)^2+homogeneousPlane(2)^2+...
    homogeneousPlane(3)^2);

distance = on/down;
roughness = distance; 
end