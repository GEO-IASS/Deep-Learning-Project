%TODO encode/decode same as predict?

load 'dataTest.mat';
load 'dataTrain.mat';


%hiddenSize1 = 100;
epochs=1000;
proportion=0.01;
reg=25;
trans='satlin';
autoenc1 = trainAutoencoder(dataTrain,100,...
    'UseGPU',true,'MaxEpochs',epochs,...
    'SparsityProportion',proportion, 'SparsityRegularization',reg,...
    'EncoderTransferFunction',trans,'DecoderTransferFunction',trans);

figure(), plotWeights(autoenc1);

reconstructed = predict(autoenc1, dataTest);

mseError = 0;
for i = 1:numel(dataTest)
    mseError = mseError + mse(double(dataTest{1, i}) - reconstructed{1, i});
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
% encoded=encode(autoenc1,dataTest);
% decoded=decode(autoenc1,encoded);
% figure;
% for i = 1:20
%     subplot(4,5,i);
%     imshow(decoded{i});
% end
