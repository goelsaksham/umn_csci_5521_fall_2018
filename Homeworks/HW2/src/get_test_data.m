function [test_X, test_y] = get_test_data(dataset_path)

% Read the whole CSV file into a matrix
input_mnist_data = dlmread(dataset_path, ',');
% Extract the test data. This corresponds to the rows with flag values = 5
test_data = input_mnist_data(input_mnist_data(:, 1) == 5, :);
% Get rid of the flag values
test_data = test_data(:, 2:end);
% Extract the X (input feature) matrix from the test data matrix
test_X = test_data(:, 2:end);
% Extract the y (labels) vector from the test data matrix
test_y = test_data(:, 1);
end