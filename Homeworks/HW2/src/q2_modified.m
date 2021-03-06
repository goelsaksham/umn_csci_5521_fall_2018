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
[principal_components, explained_var] = mypca(X, 2);
% Extract the principal components for each digit seperately
[eight_pc, ~] = get_digit_feature_matrix(principal_components, y, 8);
[nine_pc, ~] = get_digit_feature_matrix(principal_components, y, 9);

[projectionVector, ~, ~] = LDA_twoclass(vertcat(eight_pc, nine_pc), y);

projection_eight = eight_pc*projectionVector;
most_misclassified_eight = find(projection_eight == min(projection_eight));
projection_nine = nine_pc*projectionVector;
most_misclassified_nine = find(projection_nine == max(projection_nine));

% Getting the classifier vector
classifier_vector = [-projectionVector(2); projectionVector(1)];

%% Plotting Section


%% PCA Plot 
% Plot the scatter plot.
figure('Name', '2-Dimensional Space representation of Digits', 'NumberTitle', 'off');
subplot(4, 2, 1);
hold on;
scatter(eight_pc(:, 1), eight_pc(:, 2), 'r*');
scatter(nine_pc(:, 1), nine_pc(:, 2), 'b+');
xlabel('Principal Component 1');
ylabel('Principal Component 2');
title('2-Dimensional Space representation of Digits');
legend('Digit: 8','Digit: 9');
hold off;


%% PCA Plot with LDA Projection Vector and Classifier Vector
subplot(4, 2, 2);
hold on;
scatter(eight_pc(:, 1), eight_pc(:, 2), 'r*');
scatter(nine_pc(:, 1), nine_pc(:, 2), 'b+');
xl = xlim;
yl = ylim;

x_min = xl(1);
x_max = xl(2);

plot([x_min, x_max], [projectionVector(2) * x_min / projectionVector(1), projectionVector(2) * x_max / projectionVector(1)], 'k', 'LineWidth', 2);
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

scatter(projection_eight, zeros(size(projection_eight)), 'r*');
scatter(projection_nine, zeros(size(projection_nine)), 'b+');

xlabel('Projection Vector');
title('Projections of Digits onto the LDA Projection Vector');
legend('Digit: 8','Digit: 9');
hold off;


%% Digit PCA Features projected on the LDA Projection Vector
subplot(4, 2, [5 8]);
hold on;
scatter(eight_pc(:, 1), eight_pc(:, 2), 'r*', 'MarkerEdgeAlpha',.3);
scatter(nine_pc(:, 1), nine_pc(:, 2), 'b+', 'MarkerEdgeAlpha',.3);

scatter(projection_eight * projectionVector(1), projection_eight * projectionVector(2), 'r*');
scatter(projection_nine * projectionVector(1), projection_nine * projectionVector(2), 'b+');

xl = xlim;
yl = ylim;

x_min = xl(1);
x_max = xl(2);

plot([x_min, x_max], [projectionVector(2) * x_min / projectionVector(1), projectionVector(2) * x_max / projectionVector(1)], 'k', 'LineWidth', 2);
plot([x_min, x_max], [classifier_vector(2) * x_min / classifier_vector(1), classifier_vector(2) * x_max / classifier_vector(1)], '--k', 'LineWidth', 2);

xlim(xl);
ylim(yl);
xlabel('Principal Component 1');
ylabel('Principal Component 2');
title('2-Dimensional Space representation of Digits');
legend('Digit: 8','Digit: 9', 'Projection: 8', 'Projection: 9', 'Projection Vec', 'Classifier');
hold off;






figure('Name', 'Most Misclassified Nine', 'NumberTitle', 'off');
imshow(reshape(nineDigitFeatureMatrix(most_misclassified_nine, :), [28,28]));
figure('Name', 'Most Misclassified Eight', 'NumberTitle', 'off');
imshow(reshape(eightDigitFeatureMatrix(most_misclassified_eight, :), [28,28]));
%LDA

disp(most_misclassified_nine);