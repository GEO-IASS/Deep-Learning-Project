function [mse_err, accuracy] = part2_c(dataTrain, dataTest, ...
    labelsTrain, labelsTest, spar_prop, ...
    spar_regu, hiddenSize1, hiddenSize2)

    rng('default')    
    autoenc1 = trainAutoencoder(dataTrain, hiddenSize1, ...
        'MaxEpochs', 200,...
        'DecoderTransferFunction',  'logsig',   ...
        'EncoderTransferFunction',  'logsig',   ...
        'SparsityProportion',       spar_prop,  ...
        'SparsityRegularization',   spar_regu,  ...
        'UseGPU',                   true);

    feat1 = encode(autoenc1, dataTrain);

    autoenc2 = trainAutoencoder(feat1, hiddenSize2, ...
        'MaxEpochs', 200,...
        'DecoderTransferFunction',  'logsig',   ...
        'EncoderTransferFunction',  'logsig',   ...
        'SparsityProportion',       spar_prop,  ...
        'SparsityRegularization',   spar_regu,  ...
        'UseGPU',                   true);

    feat2 = encode(autoenc2, feat1);

    softnet = trainSoftmaxLayer(feat2,labelsTrain,'MaxEpochs', 200);

    deepnet = stack(autoenc1, autoenc2, softnet);

    inputSize = 28 * 28;
    xTrain = zeros(inputSize,numel(dataTrain));
    for i = 1:numel(dataTrain)
        xTrain(:,i) = dataTrain{i}(:);
    end

    % Perform fine tuning
    deepnet = train(deepnet,xTrain,labelsTrain);

    xTest = zeros(inputSize,numel(dataTest));
    for i = 1:numel(dataTest)
        xTest(:,i) = dataTest{i}(:);
    end
    y = deepnet(xTest);
%     plotconfusion(labelsTest, y);
    
    % Compute accuracy
    correct_cases = 0;
    for j = 1:numel(dataTest)
        i = 1;
        while i <= 10 && labelsTest(i, j) ~= 1
            i = i + 1;
        end
        correct_cases = correct_cases + y(i, j);
    end
    accuracy = correct_cases / numel(dataTest);


%     figure(), plotWeights(autoenc2);

    feat1_t = encode(autoenc1, dataTest);
    reconstructed_c = predict(autoenc2, feat1_t);
    reconstructed_c_d = decode(autoenc1, reconstructed_c);

    mse_err = 0;
    for i = 1:numel(dataTest)
        mse_err = mse_err + mse(double(dataTest{1, i}) - reconstructed_c_d{1, i});
    end
    mse_err = mse_err / i;
end