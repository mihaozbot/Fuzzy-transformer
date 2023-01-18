data_norm = table2array(data);
data_norm  = (data_norm  -mean(data_norm ,1))./std(data_norm ,1);
data_all = reshape((data_norm(:,4)),1,[]);
for i = 1:(size(data_norm,1)-xlength)
    data_all_class{i} = (data_norm(i:i+xlength-1,1:4))';
    data_all_class{i} = (data_norm(i:i+xlength-1,4))';
end
for i = 1:numObservations
    Xdata{i} = (data_norm((i:i+xlength-1),1:4))';
    Xdata{i} = (data_norm((i:i+xlength-1),4))';
    %Xdata{i} = reshape((data_norm(i:i+xlength-1,4)),1,[]);
    %Xdata{i} = table2array(data((i+xlength-1),1:4));
    Ydata{i} = data_norm((i+xlength+step_ahead):(i+xlength+ylength+step_ahead-1),4)';
    %Ydata{i} = reshape((data_norm((i+xlength):(i+xlength+ylength-1),4)),1,[]);
    %Ydata{i} = reshape(table2array(data((i+xlength+ylength-1),4)),[],1);
    %Ydata{i} = table2array(data((i+xlength+ylength-1),1:4));
    %Xdata{i} = table2array(data((i:i+xlength-1),1:4));
end

figure; hold on;
plot((1+step_ahead):(step_ahead+ylength),Ydata{1},'r-')
plot(1:xlength, Xdata{1},'b--')

randompermdata = randperm(numObservations);
part = [0.8,0.1,0.1];
N_parts = [1,floor(part(1)*numObservations);...
    floor(part(1)*numObservations)+1,floor((part(1)+part(2))*numObservations);...
    floor((part(1)+part(2))*numObservations)+1,numObservations];
idxTrain = randompermdata(N_parts(1,1):N_parts(1,2)); % 20% for test
idxValidation = randompermdata(N_parts(2,1):N_parts(2,2)); % remaining for training
idxTest = randompermdata(N_parts(3,1):N_parts(3,2));

if 1
XTrain = Xdata(N_parts(1,1):N_parts(1,2));
XValidation = Xdata(N_parts(2,1):N_parts(2,2));
XTest = Xdata(N_parts(3,1):N_parts(3,2));

YTrain = Ydata(N_parts(1,1):N_parts(1,2));
YValidation = Ydata(N_parts(2,1):N_parts(2,2));
YTest = Ydata(N_parts(3,1):N_parts(3,2));

else
XTrain = Xdata(idxTrain);
XValidation = Xdata(idxValidation);
XTest = Xdata(N_parts(3,1):N_parts(3,2));

YTrain = Ydata(idxTrain);
YValidation = Ydata(idxValidation);
YTest = Ydata(N_parts(3,1):N_parts(3,2));
end