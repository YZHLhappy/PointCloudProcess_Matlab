% Overview:This function is used to calculate the optimal number of leaves and tree parameters in a random forest.
% Created time: 2021.11.22
% Author: YZHLhappy
% Reference：https://zhuanlan.zhihu.com/p/367491476，Most of the programs refer to this article, I just modified it into a convenient function, thank the author of the article.
% Modify record: 2021.11.22 Create this function

function [] = FindOptimizationTreesandLeaf(Num,Type,Input_train,Output_train,MaxNumTree)
% nLeaf = Output_1; % Not added yet
% nTree = Output_2; % Not added yet

% Num = Input_1;    % Number of cycles
% (未添加） RFLeaf = Input_2;   % The default is 1, at this time: RFLeaf=[5,10,20,50,100,200,500];
%                               % If you need to modify it, you need to modify the plot part of the function by yourself.
% Type = Input_2; Type=1,classification; 
%                 Type=2,regression;
% Input_train = Input_3;  % The independent variable part of the training data
% Output_train = Input_4; % The label part of the training data (classification)
%                         % Regression part of the training data (regression)
% MaxNumTree = Input_5;   % Set the maximum number of trees (for example, 2000, this function will test the corresponding error value of 1 to 2000 trees)
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
