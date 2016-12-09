%cnnPreprocess('..\Images_Data_Clipped');

load 'dataTeststore.mat';
load 'dataTrainstore.mat';

dataTrainstoreSubset = imageDatastore('..\Images_Data_Clipped\Train\*');
dataTeststoreSubset = imageDatastore('..\Images_Data_Clipped\Test\*');

%Take 100 samples for training and 20 for testing
dataTrainstoreSubset.Files = dataTrainstore.Files(1:300);
dataTeststoreSubset.Files = dataTeststore.Files(1:20);

dataTrainstoreSubset.Labels = dataTrainstore.Labels(1:300);
dataTeststoreSubset.Labels = dataTeststore.Labels(1:20);

imageDim = 28;

layers = [imageInputLayer([imageDim imageDim]), ...
	 convolution2dLayer([9, 9],20), ...
	 averagePooling2dLayer(2), ...
     fullyConnectedLayer(10), ...
	 softmaxLayer(), ...
     classificationLayer()];
 
 %learning rate
 %momentum and decaying rate
 %batch size
 lr=0.1;
 batchSize=100;
 momentum=0.5;
 
 options = trainingOptions('sgdm', ... 
            'MaxEpochs', 25,...
            'InitialLearnRate',lr, ...
            'MiniBatchSize', batchSize, ...
            'L2Regularization',0.001,...
            'Momentum',momentum,...
            'Shuffle','once'...
             );

convnet = trainNetwork(dataTrainstoreSubset,layers,options);
YTest = classify(convnet, dataTeststoreSubset);
TTest = dataTeststoreSubset.Labels;

% convnet = trainNetwork(dataTrainstore,layers,options);
% YTest = classify(convnet, dataTeststore);
% TTest = dataTeststore.Labels;

accuracy = sum(YTest == TTest)/numel(YTest);