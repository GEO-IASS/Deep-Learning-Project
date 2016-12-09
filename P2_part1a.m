cnnPreprocess();

load 'dataTeststore.mat';
load 'dataTrainstore.mat';

dataTrainstoreSubset = imageDatastore('..\Images_Data_Clipped\Train\*');
dataTeststoreSubset = imageDatastore('..\Images_Data_Clipped\Test\*');

%Take 100 samples for training and 20 for testing
dataTrainstoreSubset.Files = dataTrainstore.Files(1:100);
dataTeststoreSubset.Files = dataTeststore.Files(1:20);

dataTrainstoreSubset.Labels = dataTrainstore.Labels(1:100);
dataTeststoreSubset.Labels = dataTeststore.Labels(1:20);

imageDim = 28;

layers = [imageInputLayer([imageDim imageDim]), ...
    convolution2dLayer([9, 9],20), ... %reluLayer(),...
    averagePooling2dLayer([2 2]), ...
    fullyConnectedLayer(10), ...
    softmaxLayer(), ...
    classificationLayer()];



lr=0.0001;
batchSize=512;
momentum=0.9;
dropFactor=0.8;

options = trainingOptions('sgdm', ...
    'MaxEpochs', 25,...
    'InitialLearnRate',lr, ...
    'MiniBatchSize', batchSize, ...
    'L2Regularization',0.001,'Momentum',momentum,...
    'LearnRateSchedule','piecewise','LearnRateDropFactor',dropFactor);

%whole set
convnet = trainNetwork(dataTrainstore,layers,options);
YTest = classify(convnet, dataTeststore);
TTest = dataTeststore.Labels;

accuracy = sum(YTest == TTest)/numel(YTest);