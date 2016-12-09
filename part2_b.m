load 'dataTest.mat';
load 'dataTrain.mat';

%Optimal values for key parameters
MaxEpochs = 1000;
SparsityProportion = 0.11;
SparsityRegularization = 22;

rng('default')
hiddenSize1 = 100;
autoenc1 = trainAutoencoder(dataTrain, hiddenSize1, ...
    'MaxEpochs',                MaxEpochs,              ...
    'DecoderTransferFunction',  'logsig',               ...
    'EncoderTransferFunction',  'logsig',               ...
    'SparsityProportion',       SparsityProportion,     ...
    'SparsityRegularization',   SparsityRegularization, ...
    'UseGPU',                   true);

figure(), plotWeights(autoenc1);
feat1 = encode(autoenc1, dataTrain);
feat1_t = encode(autoenc1, dataTest);

hiddenSize2 = 50;
autoenc2 = trainAutoencoder(feat1, hiddenSize2, ...
    'MaxEpochs',                MaxEpochs,              ...
    'DecoderTransferFunction',  'logsig',               ...
    'EncoderTransferFunction',  'logsig',               ...
    'SparsityProportion',       SparsityProportion,     ...
    'SparsityRegularization',   SparsityRegularization, ...
    'UseGPU', true);

figure(), plotWeights(autoenc2);
reconstructed_b = predict(autoenc2, feat1_t);
reconstructed_b_d = decode(autoenc1, reconstructed_b);

mse_err = 0;
for i = 1:numel(dataTest)
    mse_err = mse_err + mse(double(dataTest{1, i}) - reconstructed_b_d{1, i});
end
mse_err = mse_err / i;

figure;
for i = 1:20
    subplot(4,5,i);
    imshow(dataTest{i});
end

figure;
for i = 1:20
    subplot(4,5,i);
    imshow(uint8(reconstructed_b_d{i}));
end
