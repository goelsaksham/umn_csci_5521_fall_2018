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
[principal_components, explained_var] = pca(X, 277);
% Extract the principal components for each digit seperately
eight_pc = get_digit_feature_matrix(principal_components, y, 8);
nine_pc = get_digit_feature_matrix(principal_components, y, 9);

bestK = -1;
bestErrorRate = inf;
bestPredVec = zeros(size(y));
for k = 3:2:9
    pred_vec = zeros(size(y));
    for row_num = 1:size(principal_components, 1)
        pred_vec(row_num) = kNN(k, principal_components(row_num, :), principal_components, y);
    end
    errorRate = sum(pred_vec ~= y);
    if errorRate < bestErrorRate
        bestErrorRate = errorRate;
        bestK = k;
        bestPredVec = pred_vec;
    end
end

confMatrix = confusionmat(bestPredVec, y);