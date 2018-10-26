function [classification] = myKNN(filename, l, k)
% First load the training data
[train_X, train_y] = get_training_data(filename);
[test_X, test_y] = get_test_data(filename);
Projection_X = vertcat(train_X, test_X);
Projection_y = vertcat(train_y, test_y);

% Finding the digits for which we want the features
[eightDigitFeatureMatrix, eightVec] = get_digit_feature_matrix(Projection_X, Projection_y, 8);
[nineDigitFeatureMatrix, nineVec] = get_digit_feature_matrix(Projection_X, Projection_y, 9);

% Making a new feature matrix that contains data from only these three
% digits
X = vertcat(eightDigitFeatureMatrix, nineDigitFeatureMatrix);
y = vertcat(eightVec, nineVec);

% Run the PCA algorithm on the input feature matrix for the training set
[principal_components_system, score, explained_var] = pca(X);
principal_components = principal_components_system(:, 1:l);

% Find the projected points from the PCA
X = X - mean(X);
Projection = X * principal_components;


classification = zeros(size(y));
for row_num = 1:size(Projection, 1)
    % Running kNN algorithm on every sample in the projection matrix
    classification(row_num) = kNN(k, Projection(row_num, :), Projection, y);
end

end

