load 'dataTest.mat';
load 'dataTrain.mat';
load 'labelsTest.mat';
load 'labelsTrain.mat';

% Default values for key parameters
SparsityProportion = 0.05;
SparsityRegularization = 1;
HiddenLayerSize1 = 100;
HiddenLayerSize2 = 50;


% Experiment different sparsity proportions
mse_errs = zeros([1 21]);
accuracies = mse_errs;
params = mse_errs;
for spar_prop = 0.1:0.01:0.3
    index = int16(spar_prop / 0.01) - 9;
    params(index) = spar_prop;
    [mse_err, accuracy] = part2_c(dataTrain, dataTest, labelsTrain, ...
        labelsTest, spar_prop, SparsityRegularization, ...
        HiddenLayerSize1, HiddenLayerSize2);
    mse_errs(index) = mse_err;
    accuracies(index) = accuracy;
end
figure, plot(params, mse_errs);
figure, plot(params, accuracies);


% Experiment different sparsity regularisations
mse_errs = zeros([1 19]);
accuracies = mse_errs;
params = mse_errs;
for spar_regu = 1:0.5:10
    index = int16(spar_regu / 0.5) - 1;
    params(index) = spar_regu;
    [mse_err, accuracy] = part2_c(dataTrain, dataTest, labelsTrain, ...
        labelsTest, SparsityProportion, spar_regu, ...
        HiddenLayerSize1, HiddenLayerSize2);
    mse_errs(index) = mse_err;
    accuracies(index) = accuracy;
end
figure, plot(params, mse_errs);
figure, plot(params, accuracies);