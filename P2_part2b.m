

load 'dataTest.mat';
load 'dataTrain.mat';

%Take 100 samples for training and 20 for testing
dataTestSubset = dataTest(1, 1:20);
dataTrainSubset = dataTrain(1, 1:100);

%hiddenSize1 = 100;
epochs=1000;
proportion=0.05;
reg=1;
entrans='satlin';
detrans='satlin';
autoenc1 = trainAutoencoder(dataTrain,100,...
    'UseGPU',true,'MaxEpochs',epochs,...
    'SparsityProportion',proportion, 'SparsityRegularization',reg,...
    'EncoderTransferFunction',entrans,'DecoderTransferFunction',detrans);

features1 = encode(autoenc1,dataTrain);
autoenc2 = trainAutoencoder(features1,50,...
    'UseGPU',true,'MaxEpochs',epochs,...
    'SparsityProportion',proportion, 'SparsityRegularization',reg,...
    'EncoderTransferFunction',entrans,'DecoderTransferFunction',detrans);

% deepnet = stack(autoenc1,autoenc2);
% [deepnet,tr] = train(deepnet,dataTrain);
%
encoded=encode(autoenc1,dataTest);
decoded=predict(autoenc2,encoded);
reconstructed=decode(autoenc1,decoded);
mseError = 0;
for i = 1:numel(dataTest)
    mseError = mseError + mse(double(dataTest{:,i}) - reconstructed{:, i});
end

mseError = mseError/i;
%
figure(), plotWeights(autoenc1);
figure(), plotWeights(autoenc2);

figure;
for i = 1:20
    subplot(4,5,i);
    imshow(dataTest{i});
end

figure;
for i = 1:20
    subplot(4,5,i);
    imshow(reconstructed{i});
end

