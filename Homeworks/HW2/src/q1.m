% This script is used to answer the first part of the homework. The first
% part of the homework asks us to implement a PCA algorithm on the given
% digits dataset so that it can be reduced to 2 dimensions and then plot
% the data points on the two projected dimensions.

% First load the training data
[train_X, train_y] = get_training_data('./data/data.csv');
% Dont need Test data for this part
%[test_X, test_y] = get_test_data('./data/data.csv');

% Construct the X, and y matrices for all data
X = train_X;
y = train_y;
% Dont need Test Data for this part
% X = vertcat(train_X, test_X);
% y = vertcat(train_y, test_y);

% Finding the digits for which we want the features
[zeroDigitFeatureMatrix, zeroVec] = get_digit_feature_matrix(X, y, 0);
[eightDigitFeatureMatrix, eightVec] = get_digit_feature_matrix(X, y, 8);
[nineDigitFeatureMatrix, nineVec] = get_digit_feature_matrix(X, y, 9);
% Making a new feature matrix that contains data from only these three
% digits
X = vertcat(zeroDigitFeatureMatrix, eightDigitFeatureMatrix, nineDigitFeatureMatrix);
y = vertcat(zeroVec, eightVec, nineVec);
% Run the PCA algorithm on the input feature matrix for the training set
[projection_on_principal_components, explained_var] = mypca(X, 2);
% Extract the projection on principal components for each digit seperately
zero_pc_projection = get_digit_feature_matrix(projection_on_principal_components, y, 0);
eight_pc_projection = get_digit_feature_matrix(projection_on_principal_components, y, 8);
nine_pc_projection = get_digit_feature_matrix(projection_on_principal_components, y, 9);

% Plot the scatter plot.
hold on;
scatter(zero_pc_projection(:, 1), zero_pc_projection(:, 2), 'r*');
scatter(eight_pc_projection(:, 1), eight_pc_projection(:, 2), 'g^');
scatter(nine_pc_projection(:, 1), nine_pc_projection(:, 2), 'b+');
legend('Digit: 0','Digit: 8','Digit: 9');
xlabel('Principal Component 1');
ylabel('Principal Component 2');
title('2-Dimensional Space representation of Digits');
hold off;