% This script is responsible to run the various helper functions to answer
% part 3 of the homework.

% First load the training data
[train_X, train_y] = get_training_data('./data/data.csv');
% Finding the digits for which we want the features
[train_eightDigitFeatureMatrix, train_eightVec] = get_digit_feature_matrix(train_X, train_y, 8);
[train_nineDigitFeatureMatrix, train_nineVec] = get_digit_feature_matrix(train_X, train_y, 9);
% Making a new feature matrix that contains data from only these three
% digits
X = vertcat(train_eightDigitFeatureMatrix, train_nineDigitFeatureMatrix);
y = vertcat(train_eightVec, train_nineVec);
X = X - mean(X);
% Run the PCA algorithm on the input feature matrix for the training set.
% Keep on increasing the number of principal components on which to project 
% until the captured variance is more than 90%.
principal_component_count = 2;
[projection_on_principal_components, explained_var] = mypca(X, principal_component_count);
while explained_var < 0.9
    principal_component_count = principal_component_count + 1;
    [projection_on_principal_components, explained_var] = mypca(X, principal_component_count);
end

% Find the projections of the individual digit training data examples on
% the required number of principal components
train_eight_pc_projection = get_digit_feature_matrix(projection_on_principal_components, y, 8);
train_nine_pc_projection = get_digit_feature_matrix(projection_on_principal_components, y, 9);

% Now find the Projection Vector using LDA so that each digit can be
% projected to a 1 Dimensional surface
[LDAprojectionVector, ~, ~] = LDA_twoclass(vertcat(train_eight_pc_projection, train_nine_pc_projection), y);

train_AllProjectionsOnLDA = projection_on_principal_components * LDAprojectionVector;


% Get the 1 Dimensional Projections for each digit
train_eightDigitProjectiononLDA = train_eight_pc_projection * LDAprojectionVector;
train_most_misclassified_eight = find(train_eightDigitProjectiononLDA == min(train_eightDigitProjectiononLDA));
train_nineDigitProjectiononLDA = train_nine_pc_projection * LDAprojectionVector;
train_most_misclassified_nine = find(train_nineDigitProjectiononLDA == max(train_nineDigitProjectiononLDA));

train_trueClassLabels = y;

% Do a loop to find the best classifier:
bestClassficiationThreshold = min(train_AllProjectionsOnLDA);
train_BestErrorRate = inf;
all_ClassificationThreshold = min(train_AllProjectionsOnLDA):0.1:max(train_AllProjectionsOnLDA);
for classficiationThreshold = all_ClassificationThreshold
    % Initialize the vector which will contain all the class labels predicted
    % by LDA projection classifier.
    train_ClassLabelsPredictedbyLDA = zeros(size(train_AllProjectionsOnLDA));
    % Our classifier predicts the digits over 0 as 8 while less than 0 as 9
    train_ClassLabelsPredictedbyLDA(train_AllProjectionsOnLDA >= classficiationThreshold) = 8;
    train_ClassLabelsPredictedbyLDA(train_AllProjectionsOnLDA < classficiationThreshold) = 9;
    train_ErrorRate = sum(train_ClassLabelsPredictedbyLDA ~= train_trueClassLabels) / size(train_ClassLabelsPredictedbyLDA, 1);
    if train_ErrorRate < train_BestErrorRate
       train_BestErrorRate = train_ErrorRate;
       bestClassficiationThreshold = classficiationThreshold;
    end
end




% Constructing the Confusion Matrix for Training Data
train_AllProjectionsOnLDA = vertcat(train_eightDigitProjectiononLDA, train_nineDigitProjectiononLDA);
train_ClassLabelsPredictedbyLDA = zeros(size(train_AllProjectionsOnLDA));
% Our classifier predicts the digits over 0 as 8 while less than 0 as 9
train_ClassLabelsPredictedbyLDA(train_AllProjectionsOnLDA >= bestClassficiationThreshold) = 8;
train_ClassLabelsPredictedbyLDA(train_AllProjectionsOnLDA < bestClassficiationThreshold) = 9;

train_ConfMatrix = confusionmat(vertcat(train_eightVec, train_nineVec), train_ClassLabelsPredictedbyLDA);
train_ErrorRate = sum(train_ClassLabelsPredictedbyLDA ~= vertcat(train_eightVec, train_nineVec)) / size(train_ClassLabelsPredictedbyLDA, 1);



% Get the confusion matrix for test set
[test_X, test_y] = get_test_data('./data/data.csv');
% Finding the digits for which we want the features
[test_eightDigitFeatureMatrix, test_eightVec] = get_digit_feature_matrix(test_X, test_y, 8);
[test_nineDigitFeatureMatrix, test_nineVec] = get_digit_feature_matrix(test_X, test_y, 9);
% Making a new feature matrix that contains data from only these three
% digits
X = vertcat(test_eightDigitFeatureMatrix, test_nineDigitFeatureMatrix);
y = vertcat(test_eightVec, test_nineVec);
% Run the PCA algorithm on the input feature matrix for the training set
[test_projection_on_principal_components, test_explained_var] = mypca(X, principal_component_count);
% Extract the projection on principal components for each digit seperately
[test_eight_pc_projection, ~] = get_digit_feature_matrix(test_projection_on_principal_components, y, 8);
[test_nine_pc_projection, ~] = get_digit_feature_matrix(test_projection_on_principal_components, y, 9);

% Get the projections for each digit on the obtained surface
test_eightDigitProjectiononLDA = test_eight_pc_projection * LDAprojectionVector;
test_most_misclassified_eight = find(test_eightDigitProjectiononLDA == min(test_eightDigitProjectiononLDA));
test_nineDigitProjectiononLDA = test_nine_pc_projection * LDAprojectionVector;
test_most_misclassified_nine = find(test_nineDigitProjectiononLDA == max(test_nineDigitProjectiononLDA));


% Constructing the Confusion Matrix for test data
test_AllProjectionsOnLDA = vertcat(test_eightDigitProjectiononLDA, test_nineDigitProjectiononLDA);
test_ClassLabelsPredictedbyLDA = zeros(size(test_AllProjectionsOnLDA));
% Our classifier predicts the digits over 0 as 8 while less than 0 as 9
test_ClassLabelsPredictedbyLDA(test_AllProjectionsOnLDA >= bestClassficiationThreshold) = 8;
test_ClassLabelsPredictedbyLDA(test_AllProjectionsOnLDA < bestClassficiationThreshold) = 9;

test_ConfMatrix = confusionmat(vertcat(test_eightVec, test_nineVec), test_ClassLabelsPredictedbyLDA);
test_ErrorRate = sum(test_ClassLabelsPredictedbyLDA ~= vertcat(test_eightVec, test_nineVec)) / size(test_ClassLabelsPredictedbyLDA, 1);



% Show the images
figure('Name', 'Most Misclassified Eight', 'NumberTitle', 'off');
imshow(reshape(train_eightDigitFeatureMatrix(train_most_misclassified_eight, :), [28,28]));
figure('Name', 'Most Misclassified Nine', 'NumberTitle', 'off');
imshow(reshape(train_nineDigitFeatureMatrix(train_most_misclassified_nine, :), [28,28]));
