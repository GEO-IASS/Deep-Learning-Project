% This script is to explore the impact of hidden layer sizes
load 'dataTest.mat';
load 'dataTrain.mat';
load 'labelsTest.mat';
load 'labelsTrain.mat';

% Optimal values for key parameters
SparsityProportion = 0.3;
SparsityRegularization = 1;

params = zeros([5 5]);
mse_errs = params;
accuracies = params;
for hiddenSize1 = 80:10:120
    col = int16(hiddenSize1 / 10) - 7;
    for hiddenSize2 = 30:10:70
        row = int16(hiddenSize2 / 10) - 2;
        params(row, col) = hiddenSize2;
        [mse_err, accuracy] = part2_c(dataTrain, dataTest, labelsTrain, ...
            labelsTest, SparsityProportion, SparsityRegularization, ...
            hiddenSize1, hiddenSize2);
        mse_errs(row, col) = mse_err;
        accuracies(row, col) = accuracy;
    end
end

figure, plot(params, mse_errs);
figure, plot(params, accuracies);