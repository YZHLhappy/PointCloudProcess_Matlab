function [ptCloudPath,labelsPath,weights] = CropPointClouds_and_MergeLabels( ...
    gridSize,datasetPath,numPoints,writeFiles,numClasses)
%--------------------------------------------------------------------------
% Time:2021.12.18
% Reference function: helperCropPointCloudsAndMergeLabels.m
%                     https://www.mathworks.com/help/lidar/ug/aerial-lidar-segmentation-using-pointnet-network.html
%                     Aerial Lidar Semantic Segmentation Using PointNet++ Deep Learning
% Improvement: 1. 由于我自己的数据集是txt文件，所以我修改读取程序的部分;
%              2. 屏蔽了取label=0的点相关语句;
%              3. 由于我的点云数据不是规则图形（不论在xyz哪个视角下），因此
%                 为了防止pcdownsample(ptCloud, 'nonuniformGridSample', 6)
%                 出错（等于0后，0不能作为分母），添加if(ptCloudDense.Count<=2)
%                 的语句防止出错。
% Author:YZHLhappy
%--------------------------------------------------------------------------
% This function is used to crop and store the point cloud and it's 
% corresponding labels according to the grid sizes..
%--------------------------------------------------------------------------
% Input: gridSize,datasetPath,numPoints,writeFiles,numClasses
% gridSize: 分割网格的大小
% datasetPath：数据所在地址
% numPoints: 向下采样后的点云数量
% writeFiles：是否开启转换数据
% numClasses：点云类别的个数

% Output: ptCloudPath,labelsPath,weights
% ptCloudPath：转换后点云文件的地址
% labelsPath：转换后label文件的地址
% weights：计算后各个类别的占比
%--------------------------------------------------------------------------

weights = ones(1,numClasses);                                              % weights表示每个类别的数量，比如类别1有1000个，则weights(1) = 1000;
fileNames = dir(datasetPath);                                              % 此处表示读取输入到此函数的那个文件夹里的文件名称，并汇总所有文件名称
ptCloudPath = fullfile(datasetPath,'PointCloud');                          % 指定转换后点云数据的存放文件夹
labelsPath =  fullfile(datasetPath,'Labels');                              % 指定转换后标注数据的存放文件夹
num = 1;                                                                   % 用于给转换后的文件重命名，从1开始，每次的下一个文件夹名字+1,例如，1、2、3...
if writeFiles                                                              % 判断是否要转换点云数据，true开始转换，false不转换
    weights = zeros(1,numClasses);
    for j = 1:size(fileNames,1)                                            % 循环读取指定文件夹中的文件
        if ~endsWith(fileNames(j).name,'txt')                              % 选择文件中以.txt结尾的文件，若不是直接进入下一个for循环，若是则继续if-end后的部分
            continue;                                                      % 直接跳出for的此次循环，不执行if-end后的程序
        end
    filePath = fullfile(fileNames(j).folder,fileNames(j).name);            % filepath表示指定文件的地址和信息
    xyzL = load(filePath);                                                 % 载入此文件数据
    xyz = xyzL(:,1:3);                                                     % xyz坐标信息
    pc = pointCloud(xyz);                                                  % 构建成point cloud数据格式
    labels = uint8(xyzL(:,4));                                             % label信息

%--------------------------------------------------------------------------
% 由于我的标注中有0，所以屏蔽这几句。
% 2021.12.16
%         pc = select(pc,labels~=0);
%         labels = labels(labels~=0);
% 2021.12.18 其实也可以加上，将没有标注的数据进行剔除，因为有些数据确实不知如何标注。
%--------------------------------------------------------------------------
        weights = weights + CalculateWeights(labels,numClasses);     % 累加每个点云文件中的weights

        % Calculate the number of grids.
        numGridsY = round(diff(pc.YLimits)/gridSize(1));                   % 从X改为Y,修改原因：因为我的数据集从XY方向分割，存在很多没有点的位置，而从YZ方向会好很多
        numGridsZ = round(diff(pc.ZLimits)/gridSize(2));                   % 从Y改为Z

        % 将每一个点属于哪个grid的index计算出来
        [~,~,~,indx,indy] = histcounts2(pc.Location(:,2), ...
            pc.Location(:,3),[numGridsY,numGridsZ], ...
            'XBinLimits',pc.YLimits,'YBinLimits',pc.ZLimits);              % pc.Location(:,1),pc.Location(:,2) 改为 pc.Location(:,2),pc.Location(:,3)
                                                                           % pc.XLimits,...,pc.YLimits改为pc.YLimits,...,pc.ZLimits

        ind = sub2ind([numGridsY,numGridsZ],indx,indy);                    % 将indx和indy合并成一个线性的数组

        for i=1:numGridsY*numGridsZ                                        % 遍历每一个grid
            idx = ind==i;                                                  % 当i和ind相等时，赋值给idx，表示此i值时的点的个数
            ptCloudDense = select(pc,idx);                                 % 选择出点云pc中这些点
            labelsDense= labels(idx);                                      % 选出这些被选择点的label
%--------------------------------------------------------------------------
% 2021.12.18 由于当ptCloudDense.Count==0时，向下采样无法进行repmat操作，所以
%            我计划在此加一个if判断，判断条件：如果ptCloudDense.Count==0，则
%            直接跳过这次循环，进入for的下一次循环.
% after test:
%            经过测试ptCloudOut = pcdownsample(ptCloud,'nonuniformGridSample',6)
%            发现，当ptCloudDense.Count<=2时，会导致ptCloudOut.Count==0，因此
%            我将此if条件修改为ptCloudDense.Count<=2.
            if ptCloudDense.Count<=2
                continue;
            end
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
            [ptCloudSparse,labelsSparse] = pointCloud_DownsamplePoints( ...
                ptCloudDense,labelsDense,numPoints);                       % 向下采样点云数据，同时采样点云和对应的label
% 这么做是为了不向下采样
% 2021.12.16
% 经测试放弃，因为深度学习网络输入层是numPoints的个数）
%             ptCloudSparse = ptCloudDense;
%             labelsSparse = labelsDense;
%--------------------------------------------------------------------------
            limits = [ptCloudDense.XLimits;ptCloudDense.YLimits;...
                ptCloudDense.ZLimits];                                     % 计算采样前点云xyz三个方向的极值

            ptCloudSparseLocation = ptCloudSparse.Location;                % 采样后点云的xyz信息
            ptCloudSparseLocation(1:2,:) = limits(:,1:2)';                 % 将最小的xyz赋予采样后点云的第一个位置，最大的xyz值赋予第二个位置
            ptCloudSparseUpdated = pointCloud(ptCloudSparseLocation, ...
                'Intensity',ptCloudSparse.Intensity, ...
                'Color',ptCloudSparse.Color, ...
                'Normal',ptCloudSparse.Normal);                            % 将更改过的点云数据导入ptCloudSparseUpdated中，构建pointcloud数据结构

            ptCloudOutSparse = helperNormalizePointCloud( ...
                ptCloudSparseUpdated);
            labelsOutSparse = permute(labelsSparse,[1,3,2]);               % 重构数据维度

            if ~exist(ptCloudPath, 'dir')                                  % 新建文件夹ptCloudPath（如果之前没有的话）
                mkdir(ptCloudPath)
            end
            ptCloudSavePath = fullfile(ptCloudPath, ...                    % 存放转换后的数据
                sprintf('%03d.pcd',num));
            if ~exist(labelsPath, 'dir')                                   % 新建文件夹labelsPath（如果之前没有的话）
                mkdir(labelsPath)
            end
            labelsSavePath = fullfile(labelsPath, ...                      % 存放转换后的数据
                sprintf('%03d.png',num));

            % Save the point cloud as pcd and image in png format
            % in the given locations
            pcwrite(ptCloudOutSparse,ptCloudSavePath);                     % 保存点云数据为.pcd格式到ptCloudPath中
            imwrite(labelsOutSparse,labelsSavePath);                       % 保存点云的label为.png数据到labelsPath中
            num = num+1;
        end
    end
end
end
