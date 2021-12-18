function net = helperReplaceFunctionLayers(dagnet)
% helperReplaceFunctionLayers is used to replace the sampleAndGroup and
% fpInerpolation functionLayers present in the PointNet++ network with
% custom layers having code generation support.
%
% This is an example helper function that is subject to change or removal
% in future releases.

% Copyright 2021 MathWorks, Inc.

% Convert the dagnetwork to layerGraph.
lgraph = layerGraph(dagnet);
layers = lgraph.Layers;
inputSize = layers(1).InputSize;
pointsDim = inputSize(3);

% Iterate throught layers to find function layers.
for i = 1:size(layers)
    layer = layers(i);
    if isprop(layer,'Type') & strcmp(layer.Type,'Function')
        if strcmp(layer.Name(1:2),'SA')
            % sampleAndGroupLayer
            if layer.NumInputs == 1
                location = 'Initial';
            else
                location = 'Intermediate';
            end

            % Parse the layer properties from its description.
            layerDescription = layer.Description;
            descriptionSplit = strsplit(layerDescription);
            numClusters = str2double(descriptionSplit(2));
            radius = str2double(descriptionSplit(7));
            clusterSize = str2double(descriptionSplit(11));

            % Replace functionLayer with custom layer having code
            % generation support.
            newLayer = sampleAndGroupCodegenLayer(layer.Name,location,...
                pointsDim,numClusters,clusterSize,radius,layerDescription);
            lgraph = replaceLayer(lgraph,layer.Name,newLayer);

        elseif strcmp(layer.Name(1:2),'FP')
            % fpInterpolationLayer
            if layer.NumInputs == 4
                location = 'Intermediate';
            elseif layer.NumInputs == 3
                location = 'Final';
            end

            % Replace functionLayer with custom layer having code
            % generation support.
            newLayer = fpInterpolationCodegenLayer(layer.Name,...
                location,pointsDim);
            lgraph = replaceLayer(lgraph,layer.Name,newLayer);
        end
    end
end

% Convert the layerGraph back to dagnetwork.
net = assembleNetwork(lgraph);
save('pointnetplusCodegenNet.mat','net');
end