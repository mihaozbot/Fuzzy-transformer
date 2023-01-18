clc; clearvars -except data; close all;

%LOAD DATASET
if exist('data') == 0
    snp500
end

xlength = 100;
ylength = 10;

numObservations = size(data,1) -xlength -ylength;
for i = 1:numObservations
    Xdata(i,:,:) = table2array(data((i:i+xlength-1), 1:4));
    %Xdata{i} = table2array(data((i+xlength-1),1:4));
    Ydata(i,:) = table2array(data((i+xlength):(i+xlength+ylength-1),4));
    %Ydata{i} = table2array(data((i+xlength+ylength-1),1:4));
    %Xdata{i} = table2array(data((i:i+xlength-1),1:4));
end

randompermdata = randperm(numObservations);
part = [0.8,0.1,0.1];
N_parts = [1,floor(part(1)*numObservations);...
    floor(part(1)*numObservations)+1,floor((part(1)+part(2))*numObservations);...
    floor((part(1)+part(2))*numObservations)+1,numObservations];
idxTrain = randompermdata(N_parts(1,1):N_parts(1,2)); % 20% for test
idxValidation = randompermdata(N_parts(2,1):N_parts(2,2)); % remaining for training
idxTest = randompermdata(N_parts(3,1):N_parts(3,2));

%XTrain = Xdata(N_parts(1,1):N_parts(1,2),:,:);
%XValidation = Xdata(N_parts(2,1):N_parts(2,2),:,:);
%XTest = Xdata(N_parts(3,1):N_parts(3,2),:);

%YTrain = Ydata(N_parts(1,1):N_parts(1,2),:);
%YValidation = Ydata(N_parts(2,1):N_parts(2,2),:);
%YTest = Ydata(N_parts(3,1):N_parts(3,2),:);

XTrain = Xdata(idxTrain,:);
XValidation = Xdata(idxValidation,:);
XTest = Xdata(idxTest,:);

YTrain = Ydata(idxTrain,:);
YValidation = Ydata(idxValidation,:);
YTest = Ydata(idxTest,:);

net = newff(XTrain', YTrain', 10);
net.trainParam.epochs = 100;
net.performParam.normalization = 'standard';
net.performFcn = 'msereg';
net.trainParam.goal = 1e-4;
net.trainParam.max_fail = 100;
net.trainParam.min_grad = 1e-5;

net = train(net,XTrain',YTrain');

%VERIFIKACIJA
outputs_train  = net(XTrain');
%errors_train  = outputs_train - YTrain';
perf_train  = perform(net,outputs_train,YTrain');

%VALIDACIJA
outputs_test = net(XTest');
perf_test = perform(net, outputs_test, YTest');

outputs_all = net(reshape(Xdata,[size(Xdata,1),size(Xdata,2)*size(Xdata,3)])');

figure; hold on;
plot(YTrain())
plot(outputs_train)

figure; hold on;
plot(YTest)
plot(outputs_test)

figure; hold on;
plot(Ydata(end,:))
plot(outputs_all(end,:))

save MLP