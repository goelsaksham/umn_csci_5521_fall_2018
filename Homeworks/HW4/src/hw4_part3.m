%% Homework 4 Part 3
% This script is used to solve the Part 3 of the Homework 4 for 
% class CSCI 5521 - Introduction to Machine Learning. This part includes
% using the deltaNN function to train a neural network.

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
all_training_features = all_training_features - mean(all_training_features);
all_training_labels = vertcat(train_eight_labels, train_nine_labels);

validation_eight_features = pixels((digit_label==8) & (flags == 4), :);
validation_eight_labels = digit_label((digit_label==8) & (flags == 4), :);
validation_nine_features = pixels((digit_label==9) & (flags == 4), :);
validation_nine_labels = digit_label((digit_label==9) & (flags == 4), :);
all_validation_features = vertcat(validation_eight_features, validation_nine_features);
all_validation_features = all_validation_features - mean(all_validation_features);
all_validation_labels = vertcat(validation_eight_labels, validation_nine_labels);

test_eight_features = pixels((digit_label==8) & (flags == 5), :);
test_eight_labels = digit_label((digit_label==8) & (flags == 5), :);
test_nine_features = pixels((digit_label==9) & (flags == 5), :);
test_nine_labels = digit_label((digit_label==9) & (flags == 5), :);
all_test_features = vertcat(test_eight_features, test_nine_features);
all_test_features = all_test_features - mean(all_test_features);
all_test_labels = vertcat(test_eight_labels, test_nine_labels);


% Apply PCA
[projection_vectors,~,~,~,explained,~] = pca(all_training_features);
subspace_vectors = projection_vectors(:, 1:76);

all_training_projection_on_subspace = all_training_features * subspace_vectors;
all_validation_projection_on_subspace = all_validation_features * subspace_vectors;
all_test_projection_on_subspace = all_test_features * subspace_vectors;

training_X = all_training_projection_on_subspace';
training_y = zeros(2, size(training_X, 2));
training_y(1, all_training_labels' == 8) = 1;
training_y(2, all_training_labels' == 8) = -1;
training_y(1, all_training_labels' == 9) = -1;
training_y(2, all_training_labels' == 9) = 1;

validation_X = all_validation_projection_on_subspace';
validation_y = zeros(2, size(validation_X, 2));
validation_y(1, all_validation_labels' == 8) = 1;
validation_y(2, all_validation_labels' == 8) = -1;
validation_y(1, all_validation_labels' == 9) = -1;
validation_y(2, all_validation_labels' == 9) = 1;

test_X = all_test_projection_on_subspace';
test_y = zeros(2, size(test_X, 2));
test_y(1, all_test_labels' == 8) = 1;
test_y(2, all_test_labels' == 8) = -1;
test_y(1, all_test_labels' == 9) = -1;
test_y(2, all_test_labels' == 9) = 1;


% %% K = 18 Training
% minimal_validation_error = inf;
% rng('default');
% for random_restart = 1:10
%     k = 18;
%     rate=0.01;
%     V = randn(k, size(training_X, 1)+1);
%     W = randn(2, k+1);
%     numEpochs = 300;
%     totalError = zeros(numEpochs, 1);
%     for i = 1:numEpochs
%         current_Error = 0;
%         for sample_num = 1:size(training_X, 2)
%             x = training_X(:, sample_num);
%             t = training_y(:, sample_num);
%             [dE_dV,dE_dW,E,z,y] = deltaNN(V,W,x,t);
%             V = V - (rate * dE_dV);
%             W = W - (rate * dE_dW);
%             current_Error = current_Error + E;    
%         end
%         totalError(i) = current_Error;
%     end
%     figure(random_restart);
%     plot(totalError);
%     
%     num_validation_Error = 0;
%     for sample_num = 1:size(validation_X, 2)
%         x = validation_X(:, sample_num);
%         t = validation_y(:, sample_num);
%         [~,~,~,z,~] = deltaNN(V,W,x,t);
%         if ~is_correct_prediction(t, z)
%             num_validation_Error = num_validation_Error + 1;
%         end
%     end
%     validation_Error = num_validation_Error/size(validation_X, 2);
%     disp(validation_Error);
%     if validation_Error < minimal_validation_error
%         minimal_validation_error = validation_Error;
%     end
% end
% 
% disp(minimal_validation_error);
% disp(k);


%% K = 9 Training (Best Possible K)
rng('default');
min_validation_error = inf;
min_V = zeros(3, size(training_X, 1)+1);
min_W = zeros(2, 4);
for random_restart = 1:10
    k = 18;
    rate=0.01;
    V = randn(k, size(training_X, 1)+1);
    W = randn(2, k+1);
    numEpochs = 300;
    totalError = zeros(numEpochs, 1);
    for i = 1:numEpochs
        current_Error = 0;
        for sample_num = 1:size(training_X, 2)
            x = training_X(:, sample_num);
            t = training_y(:, sample_num);
            [dE_dV,dE_dW,E,z,y] = deltaNN(V,W,x,t);
            V = V - (rate * dE_dV);
            W = W - (rate * dE_dW);
            current_Error = current_Error + E;    
        end
        totalError(i) = current_Error;
    end
    figure(random_restart);
    plot(totalError);
    
    num_validation_Error = 0;
    for sample_num = 1:size(validation_X, 2)
        x = validation_X(:, sample_num);
        t = validation_y(:, sample_num);
        [~,~,~,z,~] = deltaNN(V,W,x,t);
        if ~is_correct_prediction(t, z)
            num_validation_Error = num_validation_Error + 1;
        end
    end
    validation_Error = num_validation_Error/size(validation_X, 2);
    if validation_Error < min_validation_error
       min_validation_error = validation_Error;
       min_V = V;
       min_W = W;
    end
    % disp(validation_Error);
    
end
num_Test_Error = 0;
for sample_num = 1:size(test_X, 2)
    x = test_X(:, sample_num);
    t = test_y(:, sample_num);
    [~, ~, ~, z, ~] = deltaNN(min_V, min_W, x, t);
    if ~is_correct_prediction(t, z)
        num_Test_Error = num_Test_Error + 1;
    end
end
num_Test_Error = num_Test_Error/size(test_X, 2);
    
disp(num_Test_Error);
%     if validation_Error < 0.1
%         break;
%     end
% disp(validation_Error);
% disp(k);
