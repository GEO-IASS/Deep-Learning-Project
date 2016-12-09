% This script is used to test different parameter configurations
load 'dataTest.mat';
load 'dataTrain.mat';

%Default values for key parameters
MaxEpochs = 100;
SparsityProportion = 0.05;
SparsityRegularization = 1;
EncoderTransferFunction = 'logsig';
DecoderTransferFunction = 'logsig';


% Test for different epoch
params = zeros([1 10]);
results = params;
for epoch = 100:100:1000
    params(epoch / 100) = epoch;
    results(epoch / 100) = part2_a(dataTrain, dataTest, epoch, ...
    SparsityProportion, SparsityRegularization, EncoderTransferFunction, ...
    DecoderTransferFunction);
end
figure;
plot(params, results);


% Test for different sparsity proportion
params = zeros([1 20]);
results = params;
for spar_prop = 0:0.01:0.19
    params(int16(spar_prop / 0.01) + 1) = spar_prop;
    results(int16(spar_prop / 0.01) + 1) = part2_a(dataTrain, dataTest,...
        MaxEpochs, spar_prop, SparsityRegularization, ...
        EncoderTransferFunction, DecoderTransferFunction);
end
figure;
plot(params, results);


% Test for different sparsity regularisation
params = zeros([1 30]);
results = params;
for spar_regu = 1:1:30
    params(spar_regu) = spar_regu;
    results(spar_regu) = part2_a(dataTrain, dataTest, MaxEpochs, ...
        SparsityProportion, spar_regu, EncoderTransferFunction, DecoderTransferFunction);
end
figure;
plot(params, results);
