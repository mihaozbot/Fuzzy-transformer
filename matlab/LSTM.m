layers = [
    sequenceInputLayer(numFeatures)
    lstmLayer(hidden_dim,'OutputMode','last') %,'OutputMode','sequence','OutputMode','last','OutputMode','last'
    dropoutLayer(dropout_probability)
    reluLayer
    fullyConnectedLayer(numChannels)
    regressionLayer
    ];

lgraph = layerGraph(layers);

if 0
    plot(lgraph)
    analyzeNetwork(lgraph)
end

