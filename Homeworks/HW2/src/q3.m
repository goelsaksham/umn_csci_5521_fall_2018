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
principal_component_count = 2;
[principal_components, explained_var] = pca(X, principal_component_count);
while explained_var < 0.9
    [principal_components, explained_var] = pca(X, principal_component_count);
    principal_component_count = principal_component_count + 1;
end

eight_pc = get_digit_feature_matrix(principal_components, y, 8);
nine_pc = get_digit_feature_matrix(principal_components, y, 9);

[a,b,LDA] = LDA_twoclass(vertcat(eight_pc, nine_pc), y);

projection_eight = eight_pc*LDA;
most_misclassified_eight = find(projection_eight == min(projection_eight));
projection_nine = nine_pc*LDA;
most_misclassified_nine = find(projection_nine == max(projection_nine));
% Plot the scatter plot.
x_min = min(min(projection_eight(:, 1)), min(projection_nine(:, 1)));
x_max = max(max(projection_eight(:, 1)), max(projection_nine(:, 1)));
figure('Name', 'Most Misclassified Nine', 'NumberTitle', 'off');
imshow(reshape(nineDigitFeatureMatrix(most_misclassified_nine, :), [28,28]));
figure('Name', 'Most Misclassified Eight', 'NumberTitle', 'off');
imshow(reshape(eightDigitFeatureMatrix(most_misclassified_eight, :), [28,28]));