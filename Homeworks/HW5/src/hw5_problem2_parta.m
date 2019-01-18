%% Homework 5 Problem 2 Part A
% Need to solve the no hidden layer neural network to recognize the marked
% area.

%% Getting the Fake Generated Data
num_samples = 500;
tr_data = (randn(2, num_samples) - [11; 6]) * 10;
tr_label = zeros(1, num_samples);
for i = 1:size(tr_label, 2)
    tr_label(1, i) = label_problem2_parta(tr_data(1, i), tr_data(2, i));
end
tr_data = tr_data / 100;

%% Training on the given data

% Initialize the training parameters
learning_rate = 0.1;
num_epochs = 2000;
V = randn(1, 3);
training_total_error = zeros(1, num_epochs);

for epoch_num = 1:num_epochs
    current_epoch_total_error = 0;
    for training_example_index = 1:size(tr_data, 2)
        % Get the training data
        x = tr_data(:, training_example_index);
        label = tr_label(1, training_example_index);
        % Do a forward and backward pass
        [dE_dV, E, z] = One_Layer_NN(V, x, label, @sigmoid, @sigmoid_derivative);
        % Update the weights
        V = V - (learning_rate * dE_dV);
        % Accumulate the Error
        current_epoch_total_error = current_epoch_total_error + E;
    end
    training_total_error(1, epoch_num) = current_epoch_total_error;
end

semilogy(training_total_error);