function [unit_projection_vector, scatter_W, scatter_Between] = LDA_twoclass(inputFeatureMatrix, targetVec)
% LDA Summary of this function goes here
%   Detailed explanation goes here
class_labels = sort(unique(targetVec));

if size(class_labels, 1) ~= 2
    error('Invalid number of classes')
end

% Getting the matrices for both the classes seperately and zero centering
% them
[digit_1_featureMatrix, ~] = get_digit_feature_matrix(inputFeatureMatrix, targetVec, class_labels(1));
digit_1_featureMatrix_zero_center = zero_center(digit_1_featureMatrix);
class_1_numTrainingExamples = size(digit_1_featureMatrix, 1);

[digit_2_featureMatrix, ~] = get_digit_feature_matrix(inputFeatureMatrix, targetVec, class_labels(2));
digit_2_featureMatrix_zero_center = zero_center(digit_2_featureMatrix);
class_2_numTrainingExamples = size(digit_2_featureMatrix, 1);

% Computing the scatter matrices for both the featue matrices
scatter_1 = digit_1_featureMatrix_zero_center' * digit_1_featureMatrix_zero_center;
scatter_2 = digit_2_featureMatrix_zero_center' * digit_2_featureMatrix_zero_center;

% Computing the within class scatter matrix:
scatter_W = scatter_1 + scatter_2;

% Computing the scatter matrices for between class
scatter_Between = (class_1_numTrainingExamples * (mean(digit_1_featureMatrix)' * mean(digit_1_featureMatrix))) + (class_2_numTrainingExamples * (mean(digit_2_featureMatrix)' * mean(digit_2_featureMatrix)));
scatter_Between = scatter_Between - ((class_1_numTrainingExamples + class_2_numTrainingExamples) * (mean(inputFeatureMatrix)' * mean(inputFeatureMatrix)));

% Finding the optimal projection vector
%[eigen_vectors, eigen_values] = eig(scatter_Between,scatter_W);
%[~, indices] = sort(diag(eigen_values), 'descend');
%eigen_vectors_sorted_by_eigen_values = eigen_vectors(:, indices);
projection_vector = scatter_W\(mean(digit_1_featureMatrix)' - mean(digit_2_featureMatrix)');
%projection_vector = eigen_vectors_sorted_by_eigen_values(:, 1);
unit_projection_vector = projection_vector/norm(projection_vector);

end
