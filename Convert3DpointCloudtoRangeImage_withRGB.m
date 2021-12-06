function [I_RGB] = Convert3DpointCloudtoRangeImage_withRGB(x,y,z,numc,numr,rgb)
% Overview: 
%   1. Use this function to convert 3D point cloud data into 2D image 
%      data. The x and y coordinates of this conversion result are the 
%      scanning angle of the scanner.
%   2. In addition, this function can also display rgb data with 2d images.
% Time: 2021.12.06
% Reference function: VahidB (2021). pointcloud2image( x,y,z,numr,numc ) 
% (https://www.mathworks.com/matlabcentral/fileexchange/55031-pointcloud2image-x-y-z-numr-numc), 
% MATLAB Central File Exchange. Retrieved December 6, 2021.
% Improvements: 
%   1. Change the x coordinate of the image to the horizontal angle of the 
%   scanner, and change the y coordinate to the vertical angle of the scanner.
%   2. Display RGB data with 2d images.
% Author: YZHLhappy

% Input: x,y,z,numc,numr,fValue
%   x: the x coordinate of the point cloud.
%   y: the y coordinate of the point cloud.
%   z: the z coordinate of the point cloud.
%   numc: desired number of rows of output image.
%   numr: desired number of rows of output image.
%   rgb: RGB three-dimensional array

% Output: I
%   I: output image with the feature values of the point cloud.

R = rgb(:,1);
G = rgb(:,2);
B = rgb(:,3);

% Convert 3D coordinates to angular coordinates
for i=1:length(x)
    r =sqrt(x(i)^2+y(i)^2);
    alpha(i) = rad2deg(atan2(x(i),y(i)));
    theta(i) = rad2deg(atan2(z(i),r));
end

alpha_min = min(alpha); alpha_max = max(alpha);
theta_min = min(theta); theta_max = max(theta);

alphaAlpha = linspace(alpha_min,alpha_max,numc);
thetaTheta = linspace(theta_min,theta_max,numr);
[ALPHA,THETA] = meshgrid(alphaAlpha,thetaTheta);
grid_centers = [ALPHA(:),THETA(:)];

class = knnsearch(grid_centers,[alpha',theta']);

local_stat = @(x)mean(x);
% R
class_stat_R = accumarray(class,R,[numr*numc 1],local_stat);
class_stat_M_R  = reshape(class_stat_R , size(ALPHA)); 
class_stat_M_R (class_stat_M_R == 0) = max(max(class_stat_M_R));
I_R = class_stat_M_R(end:-1:1,:); %  %class_stat_M(end:-1:1,end:-1:1);
I_R = ( I_R - min(min(I_R)) ) ./ ( max(max(I_R)) - min(min(I_R)) );
% I = I.*256; % If you don.t want to output grayscale colors, enable here.

% G
class_stat_G = accumarray(class,G,[numr*numc 1],local_stat);
class_stat_M_G  = reshape(class_stat_G , size(ALPHA)); 
class_stat_M_G (class_stat_M_G == 0) = max(max(class_stat_M_G));
I_G = class_stat_M_G(end:-1:1,:); %  %class_stat_M(end:-1:1,end:-1:1);
I_G = ( I_G - min(min(I_G)) ) ./ ( max(max(I_G)) - min(min(I_G)) );

% B
class_stat_B = accumarray(class,B,[numr*numc 1],local_stat);
class_stat_M_B  = reshape(class_stat_B , size(ALPHA)); 
class_stat_M_B (class_stat_M_B == 0) = max(max(class_stat_M_B));
I_B = class_stat_M_B(end:-1:1,:); %  %class_stat_M(end:-1:1,end:-1:1);
I_B = ( I_B - min(min(I_B)) ) ./ ( max(max(I_B)) - min(min(I_B)) );

% rgb three channels merged
I_RGB(:,:,1)=I_R(:,:,1); 
I_RGB(:,:,2)=I_G(:,:,1);
I_RGB(:,:,3)=I_B(:,:,1);

end