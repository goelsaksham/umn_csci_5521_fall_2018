% Load all the data
all_digits_data = dlmread('./data/Digits089.csv');
pixels = all_digits_data(:, 3:end);
flags = all_digits_data(:, 1);
digit_label = all_digits_data(:, 2);

% Get Train, Validation and Test Data
train_eight_features = pixels((digit_label==8) & (flags < 4), :);
train_eight_labels = digit_label((digit_label==8) & (flags < 4), :);
train_nine_features = pixels((digit_label==9) & (flags < 4), :);
train_nine_labels = digit_label((digit_label==9) & (flags < 4), :);
all_training_features = vertcat(train_eight_features, train_nine_features);
%tr_data = permute(reshape(all_training_features, size(all_training_features, 1), 28, 28), [2, 3, 1]);
tr_data = reshape(all_training_features, [28, 28, 1, size(all_training_features, 1)]);
all_training_labels = categorical(vertcat(train_eight_labels, train_nine_labels));

validation_eight_features = pixels((digit_label==8) & (flags == 4), :);
validation_eight_labels = digit_label((digit_label==8) & (flags == 4), :);
validation_nine_features = pixels((digit_label==9) & (flags == 4), :);
validation_nine_labels = digit_label((digit_label==9) & (flags == 4), :);
all_validation_features = vertcat(validation_eight_features, validation_nine_features);
%all_validation_features = all_validation_features - mean(all_validation_features);
%va_data = permute(reshape(all_validation_features, size(all_validation_features, 1), 28, 28), [2, 3, 1]);
va_data = reshape(all_validation_features, [28, 28, 1, size(all_validation_features, 1)]);
all_validation_labels = categorical(vertcat(validation_eight_labels, validation_nine_labels));

test_eight_features = pixels((digit_label==8) & (flags == 5), :);
test_eight_labels = digit_label((digit_label==8) & (flags == 5), :);
test_nine_features = pixels((digit_label==9) & (flags == 5), :);
test_nine_labels = digit_label((digit_label==9) & (flags == 5), :);
all_test_features = vertcat(test_eight_features, test_nine_features);
%all_test_features = all_test_features - mean(all_test_features);
%te_data = permute(reshape(all_test_features, size(all_test_features, 1), 28, 28), [2, 3, 1]);
te_data = reshape(all_test_features, [28, 28, 1, size(all_test_features, 1)]);
all_test_labels = categorical(vertcat(test_eight_labels, test_nine_labels));



layers = [ ...
    imageInputLayer([28 28 1], 'Name', 'input')
    
    convolution2dLayer(7, 3, 'Name', 'conv_1')
    batchNormalizationLayer()
    reluLayer('Name', 'relu_1')
    %maxPooling2dLayer(3)
    
    convolution2dLayer(7, 5, 'Name', 'conv_2')
    batchNormalizationLayer()
    reluLayer('Name', 'relu_2')
    %maxPooling2dLayer(3)
    
    convolution2dLayer(7, 10, 'Name', 'conv_3')
    batchNormalizationLayer()
    reluLayer('Name', 'relu_3')
    %maxPooling2dLayer(3)
    
    convolution2dLayer(7, 1, 'Name', 'conv_4')
    batchNormalizationLayer()
    reluLayer('Name', 'relu_4')
    %maxPooling2dLayer(3)
    
    fullyConnectedLayer(2, 'Name', 'fc')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classoutput')];

%% Specify training/validation options
options = trainingOptions('adam','MaxEpochs',100, ...
    'ValidationData',{va_data,all_validation_labels}, ... % imds_valid, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

%% Train the network
%net = trainNetwork(imds_train,layers,options);
net = trainNetwork(tr_data,all_training_labels,layers,options);

%% Predict the labels of new data and calculate the classification accuracy.
YPred = classify(net,te_data); % imds_test);
Ytest = all_test_labels;  % imds_test.Labels;
accuracy = sum(YPred == Ytest)/numel(Ytest) 