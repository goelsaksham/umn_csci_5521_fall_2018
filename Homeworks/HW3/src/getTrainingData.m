function [X_train, y_train] = getTrainingData(filepath)

% Load all the data from the given file
all_Digits_Data = dlmread(filepath);
% Get the data corresponding to the training data which is just records
% with flag value < 5
all_Training_Data = all_Digits_Data(all_Digits_Data(:, 1) < 5, :);
% Get just the X value/features
X_train = all_Training_Data(:, 3:end);
% Get the target/class labels
y_train = all_Training_Data(:, 2);
end

