% This script is responsible to run the various helper functions to answer
% part 2 of the homework.

% First load the training data
[train_X, train_y] = get_training_data('./data/data.csv');
% Finding the digits for which we want the features
[train_eightDigitFeatureMatrix, train_eightVec] = get_digit_feature_matrix(train_X, train_y, 8);
[train_nineDigitFeatureMatrix, train_nineVec] = get_digit_feature_matrix(train_X, train_y, 9);
% Making a new feature matrix that contains data from only these three
% digits
X = vertcat(train_eightDigitFeatureMatrix, train_nineDigitFeatureMatrix);
y = vertcat(train_eightVec, train_nineVec);
% Run the PCA algorithm on the input feature matrix for the training set
[train_projection_on_principal_components, train_explained_var] = mypca(X, 2);
% Extract the projection on principal components for each digit seperately
[train_eight_pc_projection, ~] = get_digit_feature_matrix(train_projection_on_principal_components, y, 8);
[train_nine_pc_projection, ~] = get_digit_feature_matrix(train_projection_on_principal_components, y, 9);


% Run the LDA Algorithm to project the data on 1 Dimension
[LDAprojectionVector, ~, ~] = LDA_twoclass(vertcat(train_eight_pc_projection, train_nine_pc_projection), y);

train_eightDigitProjectiononLDA = train_eight_pc_projection * LDAprojectionVector;
train_most_misclassified_eight = find(train_eightDigitProjectiononLDA == min(train_eightDigitProjectiononLDA));
train_nineDigitProjectiononLDA = train_nine_pc_projection * LDAprojectionVector;
train_most_misclassified_nine = find(train_nineDigitProjectiononLDA == max(train_nineDigitProjectiononLDA));


% Constructing the Confusion Matrix for Training Data
train_AllProjectionsOnLDA = vertcat(train_eightDigitProjectiononLDA, train_nineDigitProjectiononLDA);
train_ClassLabelsPredictedbyLDA = zeros(size(train_AllProjectionsOnLDA));
% Our classifier predicts the digits over 0 as 8 while less than 0 as 9
train_ClassLabelsPredictedbyLDA(train_AllProjectionsOnLDA >= 0) = 8;
train_ClassLabelsPredictedbyLDA(train_AllProjectionsOnLDA < 0) = 9;

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
[test_projection_on_principal_components, test_explained_var] = mypca(X, 2);
% Extract the projection on principal components for each digit seperately
[test_eight_pc_projection, ~] = get_digit_feature_matrix(test_projection_on_principal_components, y, 8);
[test_nine_pc_projection, ~] = get_digit_feature_matrix(test_projection_on_principal_components, y, 9);


test_eightDigitProjectiononLDA = test_eight_pc_projection * LDAprojectionVector;
test_most_misclassified_eight = find(test_eightDigitProjectiononLDA == min(test_eightDigitProjectiononLDA));
test_nineDigitProjectiononLDA = test_nine_pc_projection * LDAprojectionVector;
test_most_misclassified_nine = find(test_nineDigitProjectiononLDA == max(test_nineDigitProjectiononLDA));


% Constructing the Confusion Matrix
test_AllProjectionsOnLDA = vertcat(test_eightDigitProjectiononLDA, test_nineDigitProjectiononLDA);
test_ClassLabelsPredictedbyLDA = zeros(size(test_AllProjectionsOnLDA));
% Our classifier predicts the digits over 0 as 8 while less than 0 as 9
test_ClassLabelsPredictedbyLDA(test_AllProjectionsOnLDA >= 0) = 8;
test_ClassLabelsPredictedbyLDA(test_AllProjectionsOnLDA < 0) = 9;

test_ConfMatrix = confusionmat(vertcat(test_eightVec, test_nineVec), test_ClassLabelsPredictedbyLDA);
test_ErrorRate = sum(test_ClassLabelsPredictedbyLDA ~= vertcat(test_eightVec, test_nineVec)) / size(test_ClassLabelsPredictedbyLDA, 1);


% Getting the classifier vector
classifier_vector = [-LDAprojectionVector(2); LDAprojectionVector(1)];

%% Plotting Section

%% PCA Plot 
% Plot the scatter plot.
figure('Name', '2-Dimensional Space representation of Digits', 'NumberTitle', 'off');
subplot(4, 2, 1);
hold on;
scatter(train_eight_pc_projection(:, 1), train_eight_pc_projection(:, 2), 'r*');
scatter(train_nine_pc_projection(:, 1), train_nine_pc_projection(:, 2), 'b+');
xlabel('Principal Component 1');
ylabel('Principal Component 2');
title('2-Dimensional Space representation of Digits');
legend('Digit: 8','Digit: 9');
hold off;


%% PCA Plot with LDA Projection Vector and Classifier Vector
subplot(4, 2, 2);
hold on;
scatter(train_eight_pc_projection(:, 1), train_eight_pc_projection(:, 2), 'r*');
scatter(train_nine_pc_projection(:, 1), train_nine_pc_projection(:, 2), 'b+');
xl = xlim;
yl = ylim;

x_min = xl(1);
x_max = xl(2);

plot([x_min, x_max], [LDAprojectionVector(2) * x_min / LDAprojectionVector(1), LDAprojectionVector(2) * x_max / LDAprojectionVector(1)], 'k', 'LineWidth', 2);
plot([x_min, x_max], [classifier_vector(2) * x_min / classifier_vector(1), classifier_vector(2) * x_max / classifier_vector(1)], '--k', 'LineWidth', 2);

xlim(xl);
ylim(yl);
xlabel('Principal Component 1');
ylabel('Principal Component 2');
title('2-Dimensional Space representation of Digits');
legend('Digit: 8','Digit: 9', 'Projection Vec', 'Classifier');
hold off;


%% Digit PCA Features projections
subplot(4, 2, [3 4]);
hold on;

scatter(train_eightDigitProjectiononLDA, zeros(size(train_eightDigitProjectiononLDA)), 'r*');
scatter(train_nineDigitProjectiononLDA, zeros(size(train_nineDigitProjectiononLDA)), 'b+');

xlabel('Projection Vector');
title('Projections of Digits onto the LDA Projection Vector');
legend('Digit: 8','Digit: 9');
hold off;


%% Digit PCA Features projected on the LDA Projection Vector
subplot(4, 2, [5 8]);
hold on;
scatter(train_eight_pc_projection(:, 1), train_eight_pc_projection(:, 2), 'r*', 'MarkerEdgeAlpha',.3);
scatter(train_nine_pc_projection(:, 1), train_nine_pc_projection(:, 2), 'b+', 'MarkerEdgeAlpha',.3);

scatter(train_eightDigitProjectiononLDA * LDAprojectionVector(1), train_eightDigitProjectiononLDA * LDAprojectionVector(2), 'r*');
scatter(train_nineDigitProjectiononLDA * LDAprojectionVector(1), train_nineDigitProjectiononLDA * LDAprojectionVector(2), 'b+');

xl = xlim;
yl = ylim;

x_min = xl(1);
x_max = xl(2);

plot([x_min, x_max], [LDAprojectionVector(2) * x_min / LDAprojectionVector(1), LDAprojectionVector(2) * x_max / LDAprojectionVector(1)], 'k', 'LineWidth', 2);
plot([x_min, x_max], [classifier_vector(2) * x_min / classifier_vector(1), classifier_vector(2) * x_max / classifier_vector(1)], '--k', 'LineWidth', 2);

xlim(xl);
ylim(yl);
xlabel('Principal Component 1');
ylabel('Principal Component 2');
title('2-Dimensional Space representation of Digits');
legend('Digit: 8','Digit: 9', 'Projection: 8', 'Projection: 9', 'Projection Vec', 'Classifier');
hold off;



figure('Name', 'Most Misclassified Nine', 'NumberTitle', 'off');
imshow(reshape(train_nineDigitFeatureMatrix(train_most_misclassified_nine, :), [28,28]));
figure('Name', 'Most Misclassified Eight', 'NumberTitle', 'off');
imshow(reshape(train_eightDigitFeatureMatrix(train_most_misclassified_eight, :), [28,28]));
%LDA
