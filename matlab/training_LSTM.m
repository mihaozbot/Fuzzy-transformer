clc; clearvars -except data; close all; %delete(findall(0));

%LOAD DATASET
if exist('data') == 0
    snp500
end

%CHECK GPU
tbl = gpuDeviceTable(["Name","ComputeCapability",...
    "TotalMemory","MultiprocessorCount","DeviceAvailable"]);

ylength = 1;
xlength = 1000;
step_ahead = 1;
numChannels = 1;
numFeatures = 1;
hidden_dim = 100;

%LOAD MODEL
dropout_probability = 0.1;
LSTM

plot(lgraph)
numObservations = (size(data,1)-xlength-step_ahead);

dataset_snp500

options = trainingOptions("adam", ...
    SquaredGradientDecayFactor=0.99, ...
    InitialLearnRate = 1e-3, ...
    MaxEpochs = 10, ...
    MiniBatchSize = 100, ...
    Plots="training-progress", ...
    Shuffle="every-epoch",...
    ValidationData = {XValidation, [YValidation{:}]'}, ...
    ValidationFrequency = 32,...
    Verbose = false, ...
    OutputNetwork="best-validation-loss");
%    SequencePaddingDirection="left", ...
%Shuffle="every-epoch",...
%ExecutionEnvironment="gpu"
%'ValidationData',augimdsValidation, ...
%'ValidationFrequency',valFrequency, ...
%SquaredGradientDecayFactor=0.99, ...
net = trainNetwork(XTrain, [YTrain{:}]',lgraph, options);

net = resetState(net);
for i = 1:size(data_all_class,2)
    [net, outputs_all(:,i)] = predictAndUpdateState(net, data_all_class{i});
end

net = resetState(net);
for i = 1:N_parts(3,1) %size(Xdata,2)
    [net, outputs_xdata(:,i)] = predictAndUpdateState(net, Xdata{i});
end
%
%  for i = size(Xdata,2):(size(Xdata,2)+500)
%      [net, outputs_all(:,i)] = predictAndUpdateState(net, {outputs_all(i-xlength:i-1)});
%  end


for i = N_parts(3,1):(N_parts(3,1)+2000)
    [net, outputs_xdata(:,i)] = predictAndUpdateState(net, {outputs_xdata(i-xlength:i-1)});
end

save LSTM
figure; hold on;
%ydata = [Ydata{:}];
%outputs_all_end =  [outputs_all{:}];
for i = 1:1:min(size(Xdata,2),size(outputs_xdata,2))
    xdata = [Xdata{i}];
    x_data_end(i) = xdata(end);
    outputs_xdata_end(i) = outputs_xdata(end,i);
end

plot((xlength):(size(x_data_end,2)+xlength-1),x_data_end,'b')
plot((xlength+step_ahead):(length(outputs_xdata_end)+(xlength+step_ahead)-1), outputs_xdata_end,'r')
xline(N_parts(3,1))


figure; hold on;
plot(data_all,'k')
plot((xlength):(size(Xdata,2)+xlength-1),x_data_end,'b')
plot((xlength+step_ahead):(length(outputs_all)+(xlength+step_ahead)-1), outputs_all(end,:),'r')
xline(N_parts(3,1))

figure; hold on;
for i = 1:xlength:1000
    ydata = [Ydata{i}];
    xdata = [Xdata{i}];
    plot(i:i+xlength-1,xdata,'k');
    plot((i+xlength+step_ahead):(i+xlength+ylength+step_ahead-1),ydata,'b');
    plot((i+xlength+step_ahead):(i+xlength+ylength+step_ahead-1),outputs_all(:,i),'r');
    xline(i)
end
