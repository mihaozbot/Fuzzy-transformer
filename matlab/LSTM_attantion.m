layers = [
    sequenceInputLayer(numFeatures)
    %flattenLayer
    lstmLayer(hidden_dim,'OutputMode','sequence') %,'OutputMode','sequence','OutputMode','last','OutputMode','last'
    
    fullyConnectedLayer(numChannels)
    regressionLayer];

lgraph = layerGraph(layers);

if 0
    plot(lgraph)
    analyzeNetwork(lgraph)
end
