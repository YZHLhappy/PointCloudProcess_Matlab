function [] = FindOptimizationTreesandLeaf(Num,Type,Input_train,Output_train,MaxNumTree)
%% 随机森林最有叶子数和树的参数确认
% Number of Leaves and Trees Optimization
% 
% Overview:This function is used to 计算最优叶子数和树的数量。
% Created time: 2021.11.22
% Author: YZHL
% 参考：https://zhuanlan.zhihu.com/p/367491476
% Modify record: 2021.11.22 Create this function
% nLeaf = Output_1;
% nTree = Output_2;
% Num = Input_1;    % 循环的次数
% (作废） RFLeaf = Input_2; % 默认为1，此时：RFLeaf=[5,10,20,50,100,200,500];
%                   % 如果需要修改，需要使用者自行修改函数plot部分
% Type = Input_2; Type=1,classification; 
%                 Type=2,regression;
% Input_train = Input_3; % 训练数据的自变量部分
% Output_train = Input_4; % 训练数据的标签部分（classification）
%                         % 训练数据的回归值部分（regression）
% MaxNumTree = Input_5;   % 期望最大的树的数量（例如2000，此函数将测试1到2000棵树的对应精度
for RFOptimizationNum=1:Num

RFLeaf=[5,10,20,50,100,200,500];
col='rgbcmyk';
figure('Name','RF Leaves and Trees');
for i=1:length(RFLeaf)
    if(Type == 1)
        RFModel=TreeBagger(MaxNumTree,Input_train,Output_train,'Method','classification','OOBPrediction','On','MinLeafSize',RFLeaf(i));
    elseif(type == 2)
        RFModel=TreeBagger(2000,Input_train,Output_train,'Method','regression','OOBPrediction','On','MinLeafSize',RFLeaf(i));
    end
    plot(oobError(RFModel),col(i));
    hold on
end
xlabel('Number of Grown Trees');
ylabel('Mean Squared Error') ;
LeafTreelgd=legend({'5' '10' '20' '50' '100' '200' '500'},'Location','NorthEast');
title(LeafTreelgd,'Number of Leaves');
hold off;

disp(RFOptimizationNum);
end

end