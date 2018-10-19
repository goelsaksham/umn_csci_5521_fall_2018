function [mu1, mu2, S1, S2, ConfusionMatrix, ErrorRate] = classify(TrainingSet, TestSet)
% classify function fits two multivariate Gaussian Distribution on the
% given TrainingSet and then classify using the fitted Naive Bayes model
% the given TestSet and find the error rate and confusion matrix.
% 
% Solve Hw1 Q4, Student Name: Saksham Goel
%               Student ID: goelx029 | 5138568
%
% Args:
%   TrainingSet - filename of the TrainingSet (string). Contains data in
%                 CSV format
%   TestSet - filename of the TestSet (string). Contains data in CSV format
%
% Returns:
%   mu1 - The vector of means of all the individual features correspoding
%         to the class 1
%   mu2 - The vector of means of all the individual features correspoding
%         to the class 2
%   S1 - The Covariance Matrix of all the features corresponding to class 1
%   S2 - The Covariance Matrix of all the features corresponding to class 2
%   ConfusionMatrix - The Contingency/Confusion Matrix of the classifier
%                     over the Test Data. The matrix is of size 2 X 2
%                     specifying the number of elements correctly and
%                     incorrectly classified based on each class
%   ErrorRate - The percentage of test data that was incorrectly
%               classified.
%

% Loading the Training Data from the TrainingSet File to start fitting the
% classifier
TrainingData = dlmread(TrainingSet);
X_train = TrainingData(:, 1:end-1);
y_train = TrainingData(:, end);
logical_index = logical(mod(y_train, 2));
X_label_1 = X_train(logical_index, :);
X_label_2 = X_train(~logical_index, :);

% Calculating the Mean and Covariance for input features of Class 1
mu1 = mean(X_label_1);
S1 = cov(X_label_1);

% Calculating the Mean and Covariance for input features of Class 2
mu2 = mean(X_label_2);
S2 = cov(X_label_2);

% Defining the prior probabilities
prior_label_1 = 0.6;
prior_label_2 = 0.4;

% Loading the Test Data from the TestSet file to start evaluating the
% classifier
TestData = dlmread(TestSet);
X_test = TestData(:, 1:end-1);
y_test = TestData(:, end);
y_pred = zeros(size(TestData, 1), 1);

% Getting the predicted class for each test set sample
for i = 1:size(TestData, 1)
    log_c1_over_c2 = log((det(S2)/det(S1)) .^ (0.5)) + ((1/2) .* (((X_test(i, :) - mu2) / S2) * (X_test(i, :) - mu2)' )) - ((1/2) .* (((X_test(i, :) - mu1) / S1) * (X_test(i, :) - mu1)' ))  + log(prior_label_1/prior_label_2);
    if log_c1_over_c2 > 0
        y_pred(i) = 1;
    else
        y_pred(i) = 2;
    end
end

% Computing the Confusion Matrix and Error Rate
ConfusionMatrix = confusionmat(y_test, y_pred)';
ErrorRate = sum(logical(y_test ~= y_pred)) / size(y_test, 1);