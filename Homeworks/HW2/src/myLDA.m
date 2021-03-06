function [Projection, classification] = myLDA(filename, l)
%MYLDA: Function that computes the PCA projection on the data loaded from
%the given filename and then extract the data about digits 8 and 9. After
%that projects that data onto a smaller dimension subspace given by the
%argument l. After that the function projects the data onto 1 dimension
%using LDA and use a LDA Classifier on top of it.

% filename - Path to the csv file from where the data should be loaded
% l - Number of principal components on which to prject the data

% First load the training and test data
[train_X, train_y] = get_training_data(filename);
[test_X, test_y] = get_test_data(filename);
% Concatenate the data from both of these to get the whole data
X_all = vertcat(train_X, test_X);
y_all = vertcat(train_y, test_y);

% Finding the features of the required digits
[eightDigitFeatureMatrix, eightVec] = get_digit_feature_matrix(X_all, y_all, 8);
[nineDigitFeatureMatrix, nineVec] = get_digit_feature_matrix(X_all, y_all, 9);

% Making a new feature matrix that contains data from only these two
% digits. Also getting the new class label vector
X = vertcat(eightDigitFeatureMatrix, nineDigitFeatureMatrix);
y = vertcat(eightVec, nineVec);

% Normalizing the sample space
X = X - mean(X);

% Run the PCA algorithm on the input feature matrix for the training set
[Projection, ~] = mypca(X, 2);
% principal_components = principal_components_system(:, 1:2);

% We explain how we achieved this figure in the pdf attached along with the
% submission
fprintf('Variance achieved at 76 principal componenets is 0.9007 \n');

% Running the PCA algorithm on input feature matrix for training set and
% getting l principal components
[pcaProjectionofDigits, ~] = mypca(X, l);

eight_pc = get_digit_feature_matrix(pcaProjectionofDigits, y, 8);
nine_pc = get_digit_feature_matrix(pcaProjectionofDigits, y, 9);

% We get the LDA vector from this
[LDAprojectionVector, ~, ~] = LDA_twoclass(vertcat(eight_pc, nine_pc), y);

% We calculate the projection we get when we project points on LDA vector
LDA_projectionofDigits = pcaProjectionofDigits * LDAprojectionVector;

% Local function to check if projected point is more than 0 or less than
    function pred = label_prediction_fn(x)
        if x > 0
            pred = 8;
        else
            pred = 9;
        end
    end

% Matrix that classifies LDA projections
classification = arrayfun(@(x) label_prediction_fn(x), LDA_projectionofDigits);
fprintf('Error rate is %d \n', sum(classification ~= y)/size(y,1));
end

