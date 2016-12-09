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

lr=0.0001;
batchSize=512;
momentum=0.9;
dropFactor=0.8;
layer1=[20 50 70];
layer2=[20 50 70];
accuracy=zeros(3,3);
for i=1:3
    for j=1:3
        layers = [imageInputLayer([imageDim imageDim]), ...
            convolution2dLayer([5, 5],layer1(i)), ...
            averagePooling2dLayer([2 2]), ...
            convolution2dLayer([5, 5],layer2(j)), ...
            averagePooling2dLayer([2 2]), ...
            fullyConnectedLayer(10), ...
            softmaxLayer(), ...
            classificationLayer()];
        
        
        
        options = trainingOptions('sgdm', ...
            'MaxEpochs', 20,...
            'InitialLearnRate',lr, ...
            'MiniBatchSize', batchSize, ...
            'L2Regularization',0.001,'Momentum',momentum,...
            'LearnRateSchedule','piecewise','LearnRateDropFactor',dropFactor);
        
        % %subset
        % convnet = trainNetwork(dataTrainstoreSubset,layers,options);
        % YTest = classify(convnet, dataTeststoreSubset);
        % TTest = dataTeststoreSubset.Labels;
        
        %whole set
        convnet = trainNetwork(dataTrainstore,layers,options);
        YTest = classify(convnet, dataTeststore);
        TTest = dataTeststore.Labels;
        
        accuracy(i,j) = sum(YTest == TTest)/numel(YTest);
        fprintf('i:%d j:%d =%d \n',i,j,accuracy(i));
    end
end