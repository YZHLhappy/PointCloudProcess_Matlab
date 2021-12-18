function [ptCloudOut,labelsOut] = pointCloud_DownsamplePoints(ptCloud, ...
    labels,numPoints)
%--------------------------------------------------------------------------
% Time:2021.12.18
% Reference function: helperDownsamplePoints.m
%                     https://www.mathworks.com/help/lidar/ug/aerial-lidar-segmentation-using-pointnet-network.html
%                     Aerial Lidar Semantic Segmentation Using PointNet++ Deep Learning
% Improvement: 1.本计划修改此函数，但后来修改了helperCropPointCloudsAndMergeLabels
%                就取得了预期效果，因此并未进行修改，但保留了修改思路
% Author:YZHLhappy
%--------------------------------------------------------------------------
% This function is used to selects the desired number of points by
% downsampling or replicating the point cloud data.
%--------------------------------------------------------------------------
% Input: ptCloud,labels,numPoints
% ptCloud：
% labels：
% numPoints：

% Output: ptCloudOut,labelsOut
% ptCloudOut：
% labelsOut:
%--------------------------------------------------------------------------

if ceil(ptCloud.Count/numPoints)>=6
    ptCloudOut = pcdownsample(ptCloud, ...
        'nonuniformGridSample', ceil(ptCloud.Count/numPoints));
else
    ptCloudOut = pcdownsample(ptCloud, 'nonuniformGridSample', 6);
end
[~,LOCB] = ismembertol(ptCloudOut.Location,ptCloud.Location,'ByRows',true);
if ptCloudOut.Count<numPoints
    labelsOut = labels(LOCB);
    replicationFactor = ceil(numPoints/ptCloudOut.Count);                  % ptCloudOut.Count存在等于0的可能性，此时replicationFactor将无法计算，因为分母=0;
                                                                           % 修改思路：
                                                                           %        1. 在进入此函数前，当ptCloudOut.Count=0时，直接跳过并进入下一次循环。
                                                                           %           测试效果：ptCloudOut.Count<=2时，ptCloudOut.Count仍会为0，需要修改，
                                                                           %           因此前面的if条件改为：ptCloudOut.Count<=2.
                                                                           %        2. 在这个函数中修改。
    ind = repmat(1:ptCloudOut.Count,1,replicationFactor);
    ptCloudOut = select(ptCloudOut,ind(1:numPoints));
    labelsOut = labelsOut(ind(1:numPoints),:);
else
    ptCloudOut = select(ptCloud,LOCB(1:numPoints));
    labelsOut = labels(LOCB(1:numPoints));
end
end