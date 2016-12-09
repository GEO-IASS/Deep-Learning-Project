load 'dataTest.mat';
load 'dataTrain.mat';
load 'labelsTest.mat';
load 'labelsTrain.mat';

% Optimal values for key parameters
SparsityProportion = 0.3;
SparsityRegularization = 1;
HiddenSize1 = 120;
HiddenSize2 = 60;

[mse_err, accuracy] = part2_c(dataTrain, dataTest, labelsTrain, ...
            labelsTest, SparsityProportion, SparsityRegularization, ...
            HiddenSize1, HiddenSize2);