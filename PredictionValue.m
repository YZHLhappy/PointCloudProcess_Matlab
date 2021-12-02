function [Recall,Acc,F1_score,IoU,Precision] = PredictionValue(TP,TN,FP,FN)
% Time:2021.12.02
% Reference function:
% author:YZHLhappy

% This function is used to calculate the values of recall,accuracy,F1 score
% and IoU,precision.

% Input: TP,TN,FP,FN
% TP: True Positive
% FN: False Negative
% FP: False Positive
% TN: True Negative

% Output: Recall,Acc,F1_score,IoU,Precision
% Recall: sensitivity, recall, hit rate, or true positive rate (TPR)
% Acc: accuracy 
% F1_score: F1 score is the harmonic mean of precision and sensitivity
% IoU: intersection-over-union
% Precision: precision or positive predictive value (PPV)

Recall = TP/(TP+FN);                                                       % recall
Acc = (TP+TN)/(TP+TN+FP+FN);                                               % accuracy (ACC)
F1_score = 2*TP/(2*TP+FP+FN);                                              % F1 score
IoU = TP/(FP+TP+FN);                                                       % IoU
Precision = TP/(TP+FP);                                                    % precision or positive predictive value (PPV)

end