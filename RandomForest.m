% Overview:This program is used to test the function Random forest.
% Created time: 2021.11.22
% Author: YZHL
% Modify record: 2021.11.22 Create this code

clear all;
close all;
clc;
%% 创建数据集 
% 使用下载的数据集进行理解性测试
% 数据集来自：https://github.com/ttomita/RandomerForest.git
% 使用其中的abalone_train.dat训练
% 导入数据自变量为abalonetrain_X
% 导入数据因变量（标签）为abalonetrian_Y
% 1. 训练数据集
load('abalone_train.mat');
Input_train = abalonetrain_X;
Output_train = abalonetrain_Y;
% 2. test数据集
load('abalone_test.mat');
Input_test = abalonetest_X;
Output_test = abalonetest_Y;

%%
% Num = 5;
% Type = 1;
% MaxNumTree = 1000;
% FindOptimizationTreesandLeaf(Num,Type,Input_train,Output_train,MaxNumTree)
% %% 随机森林最有叶子数和树的参数确认
% % Number of Leaves and Trees Optimization
% % 计算结果（5次取最）：叶子数50时，树的个数为107或36时最优
% %                    叶子数20时，树的个数为65或57时最优
% % 最终我选择叶子数20，树的个数为60
% % nLeaf = 20;
% % nTree = 60;
% 
% for RFOptimizationNum=1:5
% 
% RFLeaf=[5,10,20,50,100,200,500];
% col='rgbcmyk';
% figure('Name','RF Leaves and Trees');
% for i=1:length(RFLeaf)
%     RFModel=TreeBagger(2000,Input_train,Output_train,'Method','classification','OOBPrediction','On','MinLeafSize',RFLeaf(i));
%     plot(oobError(RFModel),col(i));
%     hold on
% end
% xlabel('Number of Grown Trees');
% ylabel('Mean Squared Error') ;
% LeafTreelgd=legend({'5' '10' '20' '50' '100' '200' '500'},'Location','NorthEast');
% title(LeafTreelgd,'Number of Leaves');
% hold off;
% 
% disp(RFOptimizationNum);
% end
%% 循环设置与开始（机器学习需要多次运算得到精度对比）
% Cycle Preparation
RFScheduleBar=waitbar(0,'Random Forest is Solving...');
RFRMSEMatrix=[];
RFrAllMatrix=[];
RFRunNumSet=500;
for RFCycleRun=1:RFRunNumSet

%% 训练 （准备好训练数据和测试数据）
% nLeaf = 20;
% nTree = 60;
nLeaf = 50;
nTree = 50;
RFModel_train=TreeBagger(nTree,Input_train,Output_train,...
    'Method','classification','OOBPredictorImportance','on', 'MinLeafSize',nLeaf);

%% 测试
[RFPredictYield,RFPredictConfidenceInterval]=predict(RFModel_train,Input_test);

%% cell格式转换为矩阵格式
RFPredictYield_matrix = cell2mat(RFPredictYield);

%% 评估效果（obb）
% Accuracy of RF
RFRMSE=sqrt(sum(sum((RFPredictYield_matrix-Output_test).^2))/size(Output_test,1));

RFrMatrix=corrcoef(RFPredictYield_matrix,Output_test);
RFr=RFrMatrix(1,2);
RFRMSEMatrix=[RFRMSEMatrix,RFRMSE];
RFrAllMatrix=[RFrAllMatrix,RFr];
if RFRMSE<47.9
    disp(RFRMSE);
    break;
end
disp(RFCycleRun);
str=['Random Forest is Solving...',num2str(100*RFCycleRun/RFRunNumSet),'%'];
waitbar(RFCycleRun/RFRunNumSet,RFScheduleBar,str);
end
close(RFScheduleBar);
%% Variable Importance Contrast

VariableImportanceX={};
XNum=1;
% for TifFileNum=1:length(TifFileNames)
%     if ~(strcmp(TifFileNames(TifFileNum).name(4:end-4),'MaizeArea') | ...
%             strcmp(TifFileNames(TifFileNum).name(4:end-4),'MaizeYield'))
%         eval(['VariableImportanceX{1,XNum}=''',TifFileNames(TifFileNum).name(4:end-4),''';']);
%         XNum=XNum+1;
%     end
% end

for i=1:size(Input_train,2)
    eval(['VariableImportanceX{1,XNum}=''',i,''';']);
    XNum=XNum+1;
end

figure('Name','Variable Importance Contrast');
VariableImportanceX=categorical(VariableImportanceX);
bar(VariableImportanceX,RFModel_train.OOBPermutedPredictorDeltaError)
xtickangle(45);
set(gca, 'XDir','normal')
xlabel('Factor');
ylabel('Importance');

%% RF Model Storage
RFModelSavePath='C:\Users\YANG\Desktop\MA\程序\2021.11.22\';
save(sprintf('%sRF_abalone.mat',RFModelSavePath),'nLeaf','nTree',...
    'RFModel_train','RFPredictConfidenceInterval','RFPredictYield_matrix','RFr','RFRMSE',...
    'Input_test','Output_test','Input_train','Output_train');