function [I] = Convert3DpointCloudtoRangeImage_withFeaturesValue(x,y,z,numc,numr,fValue)
% Overview: 
%   1. Use this function to convert 3D point cloud data into 2D image 
%      data. The x and y coordinates of this conversion result are the 
%      scanning angle of the scanner.
%   2. In addition, the feature values of the point cloud can be 
%      displayed in the image (for example:intensity).
% Time: 2021.12.06
% Reference function: VahidB (2021). pointcloud2image( x,y,z,numr,numc ) 
% (https://www.mathworks.com/matlabcentral/fileexchange/55031-pointcloud2image-x-y-z-numr-numc), 
% MATLAB Central File Exchange. Retrieved December 6, 2021.
% Improvements: 
%   1. Change the x coordinate of the image to the horizontal angle of the 
%   scanner, and change the y coordinate to the vertical angle of the scanner.
%   2. Change from displaying only depth images to displaying any feature 
%   value (intensity etc.)
% Author: YZHLhappy

% Input: x,y,z,numc,numr,fValue
%   x: the x coordinate of the point cloud.
%   y: the y coordinate of the point cloud.
%   z: the z coordinate of the point cloud.
%   numc: desired number of rows of output image.
%   numr: desired number of rows of output image.
%   fValue: the feature values of the point cloud.

% Output: I
%   I: output image with the feature values of the point cloud.

% Convert 3D coordinates to angular coordinates
for i=1:length(x)
    r =sqrt(x(i)^2+y(i)^2);
    alpha(i) = rad2deg(atan2(x(i),y(i)));
    theta(i) = rad2deg(atan2(z(i),r));
end

alpha_l = min(alpha); yr = max(alpha);
theta_l = min(theta); zr = max(theta);

alphaAlpha = linspace(alpha_l,yr,numc);
thetaTheta = linspace(theta_l,zr,numr);
[ALPHA,THETA] = meshgrid(alphaAlpha,thetaTheta);
grid_centers = [ALPHA(:),THETA(:)];

class = knnsearch(grid_centers,[alpha',theta']);

local_stat = @(x)mean(x);
class_stat = accumarray(class,fValue,[numr*numc 1],local_stat);
class_stat_M  = reshape(class_stat , size(ALPHA)); 
class_stat_M (class_stat_M == 0) = max(max(class_stat_M));
I = class_stat_M(end:-1:1,:); %  %class_stat_M(end:-1:1,end:-1:1);
I = ( I - min(min(I)) ) ./ ( max(max(I)) - min(min(I)) );
% I = I.*256; % If you don.t want to output grayscale colors, enable here.
end