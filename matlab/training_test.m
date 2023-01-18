start = tic;

%TRAINING PARAMETERS
validationFrequency = 300;
numEpochs = 1500;
learnRate = 0.01;

figure
C = colororder;
lineLossTrain = animatedline(Color=C(2,:));
lineLossValidation = animatedline( ...
    LineStyle="--", ...
    Marker="o", ...
    MarkerFaceColor="black");
ylim([0 inf])
xlabel("Epoch")
ylabel("Loss")
grid on

% if canUseGPU
%     XTrain = gpuArray(XTrain);
% end


for epoch = 1:numEpochs

    % Evaluate the model loss and gradients.
    [loss,gradients] = dlfeval(@modelLoss,parameters,XTrain,ATrain,TTrain);

    % Update the network parameters using the Adam optimizer.
    [parameters,trailingAvg,trailingAvgSq] = adamupdate(parameters,gradients, ...
        trailingAvg,trailingAvgSq,epoch,learnRate);

    % Update the training progress plot.
    D = duration(0,0,toc(start),Format="hh:mm:ss");
    title("Epoch: " + epoch + ", Elapsed: " + string(D))
    loss = double(loss);
    addpoints(lineLossTrain,epoch,loss)
    drawnow

    % Display the validation metrics.
    if epoch == 1 || mod(epoch,validationFrequency) == 0
        YValidation = model(parameters,XValidation,AValidation);
        lossValidation = crossentropy(YValidation,TValidation,DataFormat="BC");

        lossValidation = double(lossValidation);
        addpoints(lineLossValidation,epoch,lossValidation)
        drawnow
    end

end
