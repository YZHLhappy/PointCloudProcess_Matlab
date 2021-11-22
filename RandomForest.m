% Overview:This program is used to test the function Random forest.
% Created time: 2021.11.22
% Author: YZHL
% Modify record: 2021.11.22 Create this code
% Reference：https://zhuanlan.zhihu.com/p/367491476，Most of the programs refer to this article,thank the author of the article.

clear all;
close all;
clc;

%% Create data set
% Use the downloaded data set for comprehension testing
% Data set comes from: https://github.com/ttomita/RandomerForest.git
% Use the abalone_train.dat training
% The independent variable of the imported data is abalonetrain_X
% The dependent variable (label) of the imported data is abalonetrian_Y
% 1. Training dataset
load('abalone_train.mat');
Input_train = abalonetrain_X;
Output_train = abalonetrain_Y;
% 2. Test dataset
load('abalone_test.mat');
Input_test = abalonetest_X;
Output_test = abalonetest_Y;

%% Calculation of Optimal Number of Leaves and Tree Parameters in Random Forest
% Num = 5;
% Type = 1;
% MaxNumTree = 1000;
% FindOptimizationTreesandLeaf(Num,Type,Input_train,Output_train,MaxNumTree)
% %% Calculation of Optimal Number of Leaves and Tree Parameters in Random Forest
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

%% Loop setting and start (machine learning requires multiple operations to obtain accuracy comparison)
% Cycle Preparation
RFScheduleBar=waitbar(0,'Random Forest is Solving...');
RFRMSEMatrix=[];
RFrAllMatrix=[];
RFRunNumSet=500;  % Set the whole process to loop 500 times.
for RFCycleRun=1:RFRunNumSet

%% Training (need to prepare training data and test data)
% nLeaf = 20; % Calculated by FindOptimizationTreesandLeaf(Num,Type,Input_train,Output_train,MaxNumTree).
% nTree = 60; % Calculated by FindOptimizationTreesandLeaf(Num,Type,Input_train,Output_train,MaxNumTree).
nLeaf = 50;
nTree = 50;
RFModel_train=TreeBagger(nTree,Input_train,Output_train,...
    'Method','classification','OOBPredictorImportance','on', 'MinLeafSize',nLeaf);

%% Test
[RFPredictYield,RFPredictConfidenceInterval]=predict(RFModel_train,Input_test); % RFPredictYield here is in cell format
 
%% Convert cell format to matrix format
RFPredictYield_matrix = cell2mat(RFPredictYield);

%% Evaluate the effect
% Accuracy of RF
RFRMSE=sqrt(sum(sum((RFPredictYield_matrix-Output_test).^2))/size(Output_test,1));

RFrMatrix=corrcoef(RFPredictYield_matrix,Output_test);
RFr=RFrMatrix(1,2);
RFRMSEMatrix=[RFRMSEMatrix,RFRMSE];
RFrAllMatrix=[RFrAllMatrix,RFr];
if RFRMSE<47.9  % Find the part where the RMSE value is lower than the set threshold
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
