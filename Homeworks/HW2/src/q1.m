% First load the training data
[train_X, train_y] = get_training_data('./data/data.csv');
% Finding the digits for which we want the features
[zeroDigitFeatureMatrix, zeroVec] = get_digit_feature_matrix(train_X, train_y, 0);
[eightDigitFeatureMatrix, eightVec] = get_digit_feature_matrix(train_X, train_y, 8);
[nineDigitFeatureMatrix, nineVec] = get_digit_feature_matrix(train_X, train_y, 9);
% Making a new feature matrix that contains data from only these three
% digits
X = vertcat(zeroDigitFeatureMatrix, eightDigitFeatureMatrix, nineDigitFeatureMatrix);
y = vertcat(zeroVec, eightVec, nineVec);
% Run the PCA algorithm on the input feature matrix for the training set
[principal_components, explained_var] = pca(X, 2);
% Extract the principal components for each digit seperately
zero_pc = get_digit_feature_matrix(principal_components, y, 0);
eight_pc = get_digit_feature_matrix(principal_components, y, 8);
nine_pc = get_digit_feature_matrix(principal_components, y, 9);

% Plot the scatter plot.
hold on;
scatter(zero_pc(:, 1), zero_pc(:, 2), 'r*');
scatter(eight_pc(:, 1), eight_pc(:, 2), 'g^');
scatter(nine_pc(:, 1), nine_pc(:, 2), 'b+');
legend('Digit: 0','Digit: 8','Digit: 9');
xlabel('Principal Component 1');
ylabel('Principal Component 2');
title('2-Dimensional Space representation of Digits');
hold off;