function [mAcc,mIoU] = mAccmIoU(Acc,IoU,NumLabel)
% Time:2021.12.02
% Reference function:
% author:YZHLhappy

% This function is used to calculate the values of mAcc and mIoU.

% Input: Acc,IoU,NumLabel
% Acc: accuracy of each label/class
% IoU: intersection-over-union of each label/class
% NumLabel: the numbel of label/class

% Output: mAcc,mIoU
% mAcc: mean accuracy
% mIoU: mean IoU

mAcc = sum(Acc)/NumLabel;                                                  % mean accuracy
mIoU = sum(IoU)/NumLabel;                                                  % mIoU

end