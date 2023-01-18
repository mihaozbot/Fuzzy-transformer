
numFilters = 1024;
filterSize = 5;
dropoutFactor = 0.005;
numBlocks = 2;

layer = [sequenceInputLayer(numFeatures,Normalization="zerocenter",Name="input")
        convolution1dLayer(filterSize,numFilters,DilationFactor=1,...
        Padding = "causal", Name = "conv0")];%
lgraph = layerGraph(layer);
inputname = "conv0";%"";
outputName = "conv0";%"conv0";

for i = 1:numBlocks
    dilationFactor = 2^(i-1);
    layersBranch1 = [
        convolution1dLayer(filterSize,numFilters,DilationFactor=dilationFactor,Padding="causal",Name="conv1_"+i)
        reluLayer
        layerNormalizationLayer 
        spatialDropoutLayer(dropoutFactor)
        convolution1dLayer(filterSize,numFilters,DilationFactor=dilationFactor,Padding="causal")
        layerNormalizationLayer
        reluLayer
        spatialDropoutLayer(dropoutFactor)
        additionLayer(2,Name="add_"+i)];

    % Add and connect layers.
    lgraph = addLayers(lgraph,layersBranch1);
    lgraph = connectLayers(lgraph,outputName,"conv1_"+i);

    % Skip connection.
    if  0 && (i == 1)
        % Include convolution in first skip connection.

        lgraph = addLayers(lgraph, convolution1dLayer(1,numFilters,Name="convSkip"));
        lgraph = connectLayers(lgraph,outputName, "convSkip");
        lgraph = connectLayers(lgraph, "convSkip", "add_" + i + "/in2");

    else
        lgraph = connectLayers(lgraph,outputName,"add_" + i + "/in2");

    end
    
    % Update layer output name.
    outputName = "add_" + i;
end

outputName = "add_"+numBlocks+'_2';
lgraph = addLayers(lgraph,additionLayer(2,Name=outputName));
lgraph = connectLayers(lgraph,"add_" + i,outputName + "/in1");


layersBranch2 = [
    convolution1dLayer(filterSize,numFilters,Name="conv2",...
    DilationFactor=1,Padding="causal")
    groupNormalizationLayer(128,Name="groupNorm")];%"all-channels"

%lgraph = layerGraph(layersBranch1);
lgraph = addLayers(lgraph,layersBranch2);
lgraph = connectLayers(lgraph,inputname,"conv2");
lgraph = connectLayers(lgraph,"groupNorm",outputName +"/in2");

layerheads = [
    globalMaxPooling1dLayer("Name",'gapl')
    fullyConnectedLayer(numClasses,Name="fc")
    softmaxLayer
    regressionLayer];
lgraph = addLayers(lgraph,layerheads);
lgraph = connectLayers(lgraph,outputName,"gapl");

plot(lgraph)
