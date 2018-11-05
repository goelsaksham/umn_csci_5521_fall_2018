function [X_test, y_test] = getTestData(filepath)

% Load all the data from the given file
all_Digits_Data = dlmread(filepath);
% Get the data corresponding to the test data which is just records
% with flag value = 5
all_Test_Data = all_Digits_Data(all_Digits_Data(:, 1) == 5, :);
% Get just the X value/features
X_test = all_Test_Data(:, 3:end);
% Get the target/class labels
y_test = all_Test_Data(:, 2);
end

