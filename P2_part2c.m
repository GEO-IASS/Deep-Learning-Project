

load 'dataTest.mat';
load 'dataTrain.mat';
load 'labelsTest.mat';
load 'labelsTrain.mat';




epochs=200;
proportion=0.15;%0.1-0.3
reg=5;%1-10
trans='logsig';
layer1=100;
layer2=50;
autoenc1 = trainAutoencoder(dataTrain,layer1,...
    'UseGPU',true,'MaxEpochs',epochs,...
    'SparsityProportion',proportion, 'SparsityRegularization',reg,...
    'EncoderTransferFunction',trans,'DecoderTransferFunction',trans);

features1 = encode(autoenc1,dataTrain);

autoenc2 = trainAutoencoder(features1,layer2,...
    'UseGPU',true,'MaxEpochs',epochs,...
    'SparsityProportion',proportion, 'SparsityRegularization',reg,...
    'EncoderTransferFunction',trans,'DecoderTransferFunction',trans);

features2 = encode(autoenc2,features1);
softnet = trainSoftmaxLayer(features2,labelsTrain,'LossFunction','crossentropy');

dataTrain1=zeros(784,5005);
for i=1:5005
    dataTrain1(:,i)=reshape(cell2mat(dataTrain(i)),784,1);
end

%fine tuning with training data
deepnet = stack(autoenc1,autoenc2,softnet);
deepnet= train(deepnet,dataTrain1,labelsTrain);
%

%predict testing data
dataTest1=zeros(784,1000);
for i=1:1000
    dataTest1(:,i)=reshape(cell2mat(dataTest(i)),784,1);
end

pred=deepnet(dataTest1);
plotconfusion(labelsTest,pred);

figure(), plotWeights(autoenc1);
figure(), plotWeights(autoenc2);


encoded=encode(autoenc1,dataTest);
decoded=predict(autoenc2,encoded);
reconstructed=decode(autoenc1,decoded);
mseError = 0;
for i = 1:numel(dataTest)
    mseError = mseError + mse(double(dataTest{:,i}) - reconstructed{:, i});
end
mseError = mseError/i;

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

