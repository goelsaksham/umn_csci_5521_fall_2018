% This script is responsible to run the various helper functions to answer
% part 4 of the homework.


% First load the training data
[train_X, train_y] = get_training_data('./data/data.csv');
% Finding the digits for which we want the features
[train_eightDigitFeatureMatrix, train_eightVec] = get_digit_feature_matrix(train_X, train_y, 8);
[train_nineDigitFeatureMatrix, train_nineVec] = get_digit_feature_matrix(train_X, train_y, 9);
% Making a new feature matrix that contains data from only these three
% digits
X_train = vertcat(train_eightDigitFeatureMatrix, train_nineDigitFeatureMatrix);
y_train = vertcat(train_eightVec, train_nineVec);

% Run the PCA algorithm on the input feature matrix for the training set
principal_component_count = 2;
[~, train_explained_var] = mypca(X_train, principal_component_count);
while train_explained_var < 0.9
    principal_component_count = principal_component_count + 1;
    [~, train_explained_var] = mypca(X_train, principal_component_count);
end

% Run the PCA algorithm on the input feature matrix for the training set
% with the required number of principal components and get the projections
[train_principal_components_projections, train_explained_var] = mypca(X_train, principal_component_count);

bestK = -1;
train_bestErrorRate = inf;
train_bestPredVec = zeros(size(y_train));
for k = 1:2:9
    train_pred_vec = zeros(size(y_train));
    for row_num = 1:size(train_principal_components_projections, 1)
        train_pred_vec(row_num) = kNN(k, train_principal_components_projections(row_num, :), train_principal_components_projections, y_train);
    end
    errorRate = sum(train_pred_vec ~= y_train);
    if errorRate < train_bestErrorRate
        train_bestErrorRate = errorRate;
        bestK = k;
        train_bestPredVec = train_pred_vec;
    end
end

train_ConfMatrix = confusionmat(y_train, train_bestPredVec);
train_ErrorRate = sum(y_train ~= train_bestPredVec) / size(train_bestPredVec, 1);




% First load the test data
[test_X, test_y] = get_test_data('./data/data.csv');
% Finding the digits for which we want the features
[test_eightDigitFeatureMatrix, test_eightVec] = get_digit_feature_matrix(test_X, test_y, 8);
[test_nineDigitFeatureMatrix, test_nineVec] = get_digit_feature_matrix(test_X, test_y, 9);
% Making a new feature matrix that contains data from only these three
% digits
X_test = vertcat(test_eightDigitFeatureMatrix, test_nineDigitFeatureMatrix);
y_test = vertcat(test_eightVec, test_nineVec);

% Run the PCA algorithm on the input feature matrix for the training set
% with the required number of principal components and get the projections
[test_principal_components_projections, test_explained_var] = mypca(X_test, principal_component_count);

test_pred_vec = zeros(size(y_test));
for row_num = 1:size(test_principal_components_projections, 1)
    test_pred_vec(row_num) = kNN(bestK, test_principal_components_projections(row_num, :), train_principal_components_projections, y_train);
end

test_ConfMatrix = confusionmat(y_test, test_pred_vec);
test_ErrorRate = sum(y_test ~= test_pred_vec) / size(test_pred_vec, 1);