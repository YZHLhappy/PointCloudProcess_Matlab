function [TP,TN,FP,FN] = calTNTPFPFN(label_ref,label_pre,label)
% time:2021.12.02
% Reference function:Jason Joseph Rebello (2021). True Positives, False Positives, 
% True Negatives, False Negatives from 2 Matrices 
% (https://www.mathworks.com/matlabcentral/fileexchange/47364-true-positives-false-positives-true-negatives-false-negatives-from-2-matrices), 
% MATLAB Central File Exchange. Retrieved December 2, 2021.
% Improvement: Expand the value range to any value (no longer just 0 and 1)
% author:YZHLhappy

% This function is used to calculate the values of TP, TN, FP, FN.

% Input: label_ref,label_pre,label
% label_ref: the value of the reference label
% label_pre: the value of the predicted label
% label: the label used as a benchmark

% Output: TP,TN,FP,FN
% TP: True Positive
% FN: False Negative
% FP: False Positive
% TN: True Negative


for i=1:length(label_ref)
    if (label_ref(i)==label)
        label_ref(i) = 1;
    else
        label_ref(i) = 0;
    end
end

for i=1:length(label_pre)
    if (label_pre(i)==label)
        label_pre(i) = 1;
    else
        label_pre(i) = 0;
    end
end

adder = label_ref + label_pre;
TP = length(find(adder == 2));
TN = length(find(adder == 0));
subtr = label_ref - label_pre;
FP = length(find(subtr == -1));
FN = length(find(subtr == 1));

end