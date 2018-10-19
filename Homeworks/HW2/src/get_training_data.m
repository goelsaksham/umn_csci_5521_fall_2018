function [train_X, train_y] = get_training_data(dataset_path)

% Read the whole CSV file into a matrix
input_mnist_data = dlmread(dataset_path, ',');
% Extract the training data. This corresponds to the rows with flag values in
% range 1 to 4 (inclusive)
training_data = input_mnist_data(input_mnist_data(:, 1) < 5, :);
% Get rid of the flag values
training_data = training_data(:, 2:end);
% Extract the X (input feature) matrix from the training data matrix
train_X = training_data(:, 2:end);
% Extract the y (labels) vector from the training data matrix
train_y = training_data(:, 1);
end

