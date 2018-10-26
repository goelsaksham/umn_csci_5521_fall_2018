function [Projection, classification] = myLDA(filename, l)
%MYLDA: Function that computes the PCA projection and the LDA
%classification

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
[principal_components_system, score, explained_variance] = pca(X);
principal_components = principal_components_system(:, 1:2);

sum_of_variance = 0;
count = 1;
while sum_of_variance < 0.9
    sum_of_variance = sum_of_variance + explained_variance(count,1);
    count = count + 1;
end

disp(explained_variance);

X = X - mean(X);
Projection = X * principal_components;


% Running the PCA algorithm on input feature matrix for training set and
% getting l principal components
principal_components_for_LDA = principal_components_system(:, 1:l);
Projection_for_LDA = X * principal_components_for_LDA;

eight_pc = get_digit_feature_matrix(Projection_for_LDA, y, 8);
nine_pc = get_digit_feature_matrix(Projection_for_LDA, y, 9);

% We get the LDA vector from this
[LDA, a, b] = LDA_twoclass(vertcat(eight_pc, nine_pc), y);

% We calculate the projection we get when we project points on LDA vector
LDA_projection = Projection_for_LDA * LDA;

% Local function to check if projected point is more than 0 or less than
    function pred = label_prediction_fn(x)
        if x > 0
            pred = 8;
        else
            pred = 9;
        end
    end

% Matrix that classifies LDA projections
classification = arrayfun(@(x) label_prediction_fn(x), LDA_projection);
sum(classification ~= y)
end

