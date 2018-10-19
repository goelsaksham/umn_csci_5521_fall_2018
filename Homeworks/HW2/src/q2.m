% First load the training data
[train_X, train_y] = get_training_data('./data/data.csv');
% Finding the digits for which we want the features
[eightDigitFeatureMatrix, eightVec] = get_digit_feature_matrix(train_X, train_y, 8);
[nineDigitFeatureMatrix, nineVec] = get_digit_feature_matrix(train_X, train_y, 9);
% Making a new feature matrix that contains data from only these three
% digits
X = vertcat(eightDigitFeatureMatrix, nineDigitFeatureMatrix);
y = vertcat(eightVec, nineVec);
% Run the PCA algorithm on the input feature matrix for the training set
[principal_components, explained_var] = pca(X, 2);
% Extract the principal components for each digit seperately
eight_pc = get_digit_feature_matrix(principal_components, y, 8);
nine_pc = get_digit_feature_matrix(principal_components, y, 9);

% Plot the scatter plot.
hold on;
scatter(eight_pc(:, 1), eight_pc(:, 2), 'r*');
scatter(nine_pc(:, 1), nine_pc(:, 2), 'b+');
legend('Digit: 8','Digit: 9');
xlabel('Principal Component 1');
ylabel('Principal Component 2');
title('2-Dimensional Space representation of Digits');
hold off;