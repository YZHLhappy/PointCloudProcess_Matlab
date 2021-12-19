%--------------------------------------------------------------------------
% Time:2021.12.18
% Reference function: 
%    https://www.mathworks.com/help/lidar/ug/aerial-lidar-segmentation-using-pointnet-network.html
%    Aerial Lidar Semantic Segmentation Using PointNet++ Deep Learning
% Improvement: 1. 由于我自己的数据集是txt文件，所以我修改读取程序的部分;
%              2. 屏蔽了取label=0的点相关语句;
%              3. 由于我的点云数据不是规则图形（不论在xyz哪个视角下），因此
%                 为了防止pcdownsample(ptCloud, 'nonuniformGridSample', 6)
%                 出错（等于0后，0不能作为分母），添加if(ptCloudDense.Count<=2)
%                 的语句防止出错。
% Author:YZHLhappy
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
% 这个程序是用来构建我自己的pointNet++程序
% 这个程序包含载入数据，分割数据，训练模型，测试模型，共4个部分
% 1. 载入数据
% 2. 分割数据（采样数据）
% 3. 训练模型
% 4. 测试模型（这其中包含：测试数据的读取，分割测试数据，测试三个部分）

%--------------------------------------------------------------------------
%                             1. 载入数据
%--------------------------------------------------------------------------
% 1. 载入训练数据的地址
dataFolder = fullfile('C:\Users\YANG\Desktop\2021.12.15\data_19');         % dataFolder表示存放数据的文件夹地址
trainDataFolder = fullfile(dataFolder,'train');                            % 存放训练数据的文件夹地址
testDataFolder = fullfile(dataFolder,'test');                              % 存放测试数据的文件夹地址

% 1.1. 类别的名称，最初测试是ground和vegetation两个，后期将加上buildings和
%      low vegetation
classNames = [
               "ground" 
               "vegetation"
             ];
%--------------------------------------------------------------------------
%                      2. 分割数据，转换并保存数据
%--------------------------------------------------------------------------
% 2. 分割数据，由于数据很大，内存会有较大压力，所以此处将数据分割成较小的且不
%    重叠的数据块.
% 2.1. 分割数据成不重叠的几个部分，大小设置根据需求，此处设置为10x10 
%     （后期看看能不能有其他方式来处理，已解决，现在可以设置为任意值）.
% 2.2. 向下采样，使点云变成一个固定的大小（因为后面构建pointNet++神经网络时需
%      要这个参数，因此此处需要一个固定值）
% 2.3. 将点云范围normalize到[0,1]范围内.
% 2.4. 将裁剪后的网格和语义标签分别保存为 PCD 和 PNG 文件.

% 2.1. 分割数据成不重叠的几个部分,参数设置
gridSize = [10,10];                                                        % 分割块的大小
numPoints = 5000;                                                          % 乡下采样后的点数量

% 设定一个判断，表示是否需要转换数据，判断变量writeFiles
writeFiles = true;                                                         % true: 需要转换，false：不需要转换
numClasses = numel(classNames);                                            % 计算类别的数量

% 2.2. 分割数据并向下采样点云
% 2.4. 将裁剪后的网格和语义标签分别保存为 PCD 和 PNG 文件
[pcCropTrainPath,labelsCropTrainPath,weights] = ...
    CropPointClouds_and_MergeLabels(...
    gridSize,trainDataFolder,numPoints,writeFiles,numClasses);             % 用于分割点云并将label和点云合并

% 2.3. 将点云范围normalize到[0,1]范围内
[maxWeight,maxLabel] = max(weights);
weights = sqrt(maxWeight./weights);

% 创建一个 fileDatastore 对象以使用 pcread 函数加载 PCD 文件。
ldsTrain = fileDatastore(pcCropTrainPath,'ReadFcn',@(x) pcread(x));

% 指定从 1 到类数的标签 ID。
labelIDs = 1 : numClasses;
pxdsTrain = pixelLabelDatastore(labelsCropTrainPath,classNames,labelIDs);

% use the helperConvertPointCloud function, to convert the point cloud to 
% cell array.
ldsTransformed = transform(ldsTrain,@(x) helperConvertPointCloud(x));

% Use the combine function to combine the point clouds and pixel labels 
% into a single datastore for training.
dsTrain = combine(ldsTransformed,pxdsTrain);

%--------------------------------------------------------------------------
%                            pointNet++架构
%--------------------------------------------------------------------------
% Define the PointNet++ architecture using the pointnetplusLayers function.
lgraph = pointnetplusLayers(numPoints,3,numClasses);
% Replace the FocalLoss layer with pixelClassificationLayer.
larray = pixelClassificationLayer('Name','SegmentationLayer', ...
    'ClassWeights', weights,'Classes',classNames);
lgraph = replaceLayer(lgraph,'FocalLoss',larray);
% Use the Adam optimization algorithm to train the network.
learningRate = 0.0005;
l2Regularization = 0.01;
numEpochs = 1;
miniBatchSize = 6;
learnRateDropFactor = 0.1;
learnRateDropPeriod = 10;
gradientDecayFactor = 0.9;
squaredGradientDecayFactor = 0.999;

options = trainingOptions('adam', ...
    'InitialLearnRate',learningRate, ...
    'L2Regularization',l2Regularization, ...
    'MaxEpochs',numEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',learnRateDropFactor, ...
    'LearnRateDropPeriod',learnRateDropPeriod, ...
    'GradientDecayFactor',gradientDecayFactor, ...
    'SquaredGradientDecayFactor',squaredGradientDecayFactor, ...
    'Plots','training-progress');
%--------------------------------------------------------------------------
%                       3. Train Model pointNet++
%--------------------------------------------------------------------------
% Train Model
tic
doTraining = true;                                                         % doTraining==true: 要训练
                                                                           % doTraining==false: 使用训练好的pointNet++模型
if doTraining
    % Train the network on the dsTrain datastore using the trainNetwork
    % function.
    [net, info] = trainNetwork(dsTrain,lgraph,options);
else
    % Load the pretrained network.
    load(['C:\Users\YANG\Desktop\2021.12.15\data_19\...' ...
        'pointNetplus_model_12.17_21Uhr.mat'],'net');
end
toc
%%
%--------------------------------------------------------------------------
%                            4. Test Model
% 4.1. 设置分割点云和标注的格式
numNearestNeighbors = 20;                                                  % 由于点云分割是向下采样的结果，需要将采样后的数据映射回原点云，
                                                                           % 此处设置的值将标签分给这个范围内的点。
radius = 0.05;

% aa = load("C:\Users\YANG\Desktop\2021.12.15\data_19\test\test_2.txt");
aa = load("C:\Users\YANG\Desktop\Label(from 12.16)\测试使用\2.txt");       % 载入测试的数据
pc = pointCloud(aa(:,1:3));                                                % 点云的xyz数据
l = uint8(aa(:,4));                                                        % 点云的label                                              
labelsDenseTarget = l;

% Select only labeled data.
pc = select(pc,labelsDenseTarget~=0);
labelsDenseTarget = labelsDenseTarget(labelsDenseTarget~=0);

% Initialize prediction labels 
labelsDensePred = zeros(size(labelsDenseTarget));                          % 初始化数组，有利于程序运行效率

% Calculate the number of non-overlapping grids based on gridSize, XLimits,
% and YLimits of the point cloud.
% 分割测试数据
numGridsX = round(diff(pc.YLimits)/gridSize(1));                           % 从X改为Y, 修改分割程序后，需要考虑此处要不要该回去（2021.12.18），未测试。
numGridsY = round(diff(pc.ZLimits)/gridSize(2));                           % 从Y改为Z

[~,edgesX,edgesY,indx,indy] = histcounts2(pc.Location(:,2),...             % pc.Location(:,1),pc.Location(:,2) 改为 pc.Location(:,2),pc.Location(:,3)
    pc.Location(:,3), [numGridsX,numGridsY],'XBinLimits',pc.YLimits,...    % pc.XLimits,...,pc.YLimits 改为 pc.YLimits,...,pc.ZLimits
    'YBinLimits',pc.ZLimits);                                              

ind = sub2ind([numGridsX,numGridsY],indx,indy);

% Iterate over all the non-overlapping grids and predict the labels using 
% the semanticseg function.
for num=1:numGridsX*numGridsY
    idx = ind==num;
    ptCloudDense = select(pc,idx);
    labelsDense = labelsDenseTarget(idx);

    % Use the helperDownsamplePoints function, attached to this example as
    % a supporting file, to extract a downsampled point cloud from the
    % dense point cloud.
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
% 2021.12.17 屏蔽，从而不向下采样
%     测试后：恢复，因为神经网络输入设置要定值
    ptCloudSparse = pointCloud_DownsamplePoints(ptCloudDense, ...
        labelsDense,numPoints);
%--------------------------------------------------------------------------
    % Make the spatial extent of the dense point cloud and the sparse point
    % cloud same.
    limits = [ptCloudDense.XLimits;ptCloudDense.YLimits;ptCloudDense.ZLimits];
    ptCloudSparseLocation = ptCloudSparse.Location;
    ptCloudSparseLocation(1:2,:) = limits(:,1:2)';
    ptCloudSparse = pointCloud(ptCloudSparseLocation,'Color',ptCloudSparse.Color, ...
        'Intensity',ptCloudSparse.Intensity, ...
        'Normal',ptCloudSparse.Normal);

    % Use the helperNormalizePointCloud function, attached to this example as
    % a supporting file, to normalize the point cloud between 0 and 1.
    ptCloudSparseNormalized = helperNormalizePointCloud(ptCloudSparse);
    ptCloudDenseNormalized = helperNormalizePointCloud(ptCloudDense);

    % Use the helperConvertPointCloud function, defined at the end of this
    % example, to convert the point cloud to a cell array and to permute the
    % dimensions of the point cloud to make it compatible with the input layer
    % of the network.
    ptCloudSparseForPrediction = helperConvertPointCloud(ptCloudSparseNormalized);

    % Get the output predictions.
    labelsSparsePred = semanticseg(ptCloudSparseForPrediction{1,1}, ...
        net,'OutputType','uint8');

    % Use the helperInterpolate function, attached to this example as a
    % supporting file, to calculate labels for the dense point cloud,
    % using the sparse point cloud and labels predicted on the sparse point
    % cloud.
    interpolatedLabels = helperInterpolate(ptCloudDenseNormalized, ...     % 将预测的标签扩散到附近点云（因为预测的是采样后的）
        ptCloudSparseNormalized,labelsSparsePred,numNearestNeighbors, ...
        radius,maxLabel,numClasses);

    labelsDensePred(idx) = interpolatedLabels;
end

% Evaluate Network
confusionMatrix = segmentationConfusionMatrix(labelsDensePred, ...
    double(labelsDenseTarget),'Classes',1:numClasses);
metrics = evaluateSemanticSegmentation({confusionMatrix},classNames,'Verbose',false);
metrics.DataSetMetrics
metrics.ClassMetrics

% 显示结果（有待调整，调整成更好的视觉效果和标注，题目等信息）
pcshow(pc.Location, labelsDensePred);


















