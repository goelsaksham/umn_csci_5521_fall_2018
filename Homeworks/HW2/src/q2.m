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

[a,b,LDA] = LDA_twoclass(vertcat(eight_pc, nine_pc), y);

projection_eight = eight_pc*LDA;
most_misclassified_eight = find(projection_eight == min(projection_eight));
projection_nine = nine_pc*LDA;
most_misclassified_nine = find(projection_nine == max(projection_nine));
% Plot the scatter plot.
classifier = [-LDA(2); LDA(1)];
x_min = min(min(projection_eight(:, 1)), min(projection_nine(:, 1)));
x_max = max(max(projection_eight(:, 1)), max(projection_nine(:, 1)));
figure('Name', 'Most Misclassified Nine', 'NumberTitle', 'off');
imshow(reshape(nineDigitFeatureMatrix(most_misclassified_nine, :), [28,28]));
figure('Name', 'Most Misclassified Eight', 'NumberTitle', 'off');
imshow(reshape(eightDigitFeatureMatrix(most_misclassified_eight, :), [28,28]));
figure('Name', '2-Dimensional Space representation of Digits', 'NumberTitle', 'off');
hold on;
% scatter(LDA);
scatter(eight_pc(:, 1), eight_pc(:, 2), 'r*');
scatter(nine_pc(:, 1), nine_pc(:, 2), 'b+');
scatter(projection_eight * LDA(1), projection_eight * LDA(2), 'g');
scatter(projection_nine * LDA(1), projection_nine * LDA(2), 'm');
legend('Digit: 8','Digit: 9');
plot([x_min, x_max], [classifier(2) * x_min / classifier(1), classifier(2) * x_max / classifier(1)],'k');
xlabel('Principal Component 1');
ylabel('Principal Component 2');
title('2-Dimensional Space representation of Digits');
hold off;
%LDA
