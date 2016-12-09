% This script shows the two sets of optimal parameter configurations
load 'dataTest.mat';
load 'dataTrain.mat';

% Optimal values for key parameters
MaxEpochs = 1000;
SparsityProportion = 0.11;
SparsityRegularization = 22;
EncoderTransferFunction = 'logsig';
DecoderTransferFunction = 'logsig';

% logsig-logsig transfer functions
[mse_err_log, log_en, log_de] = part2_a(dataTrain, dataTest, MaxEpochs, ...
    SparsityProportion, SparsityRegularization, ...
    'logsig', 'logsig');
figure;
for i = 1:20
    subplot(4,5,i);
    imshow(uint8(log_de{i}));
end


% satlin-satlin transfer functions
[mse_err_sat, sat_en, sat_de] = part2_a(dataTrain, dataTest, MaxEpochs, ...
    SparsityProportion, SparsityRegularization, ...
    'satlin', 'satlin');
figure;
for i = 1:20
    subplot(4,5,i);
    imshow(sat_de{i});
end


% plot the test images
figure;
for i = 1:20
    subplot(4,5,i);
    imshow(dataTest{i});
end