function [mseError, encoded, decoded] = part2_a(dataTrain, dataTest, ...
    max_epoch, spar_prop, spar_regu, en_trans, de_trans)

    hiddenSize = 100;
    autoenc = trainAutoencoder(dataTrain, hiddenSize, ...
        'MaxEpochs',                max_epoch, ...
        'DecoderTransferFunction',  de_trans,  ...
        'EncoderTransferFunction',  en_trans,  ...
        'SparsityProportion',       spar_prop, ...
        'SparsityRegularization',   spar_regu, ...
        'UseGPU',                   true);

    figure(), plotWeights(autoenc);
    
    encoded = encode(autoenc, dataTest);
    decoded = decode(autoenc, encoded);
    
    mseError = 0;
    for i = 1:numel(dataTest)
        mseError = mseError + mse(double(dataTest{1, i}) - decoded{1, i});
    end
    mseError = mseError / i;
end



