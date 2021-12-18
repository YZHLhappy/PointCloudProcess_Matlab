function net = getPointnetplusNet
% This function returns a trained DAGNetwork object for PointNet++ network.

% Copyright 2021 The Mathworks, Inc.

pretrainedPointNetPlusNet = "pointnetplusTrained.mat";
pointnetplusNet = load(pretrainedPointNetPlusNet,'net');
net = pointnetplusNet.net;
end

