function [principal_components, explained_var] = pca(feature_matrix, num_principal_components)
% Make sure that the input feature_matrix is zero centered
zeroCenteredFeatureMatrix = zero_center(feature_matrix);
% Performing a Single Value Decomposition using the inbuilt svd function
% over the feature matrix
[U, Sigma, ~] = svd(zeroCenteredFeatureMatrix);
% Finding the required number of principal components
% First check whether the given argument for the number of principal
% components is possible or not

if size(Sigma, 2) < num_principal_components || size(Sigma, 1) < num_principal_components || num_principal_components < 1
    error('Incorrect number of principal components provided. Number of Principal Components required: %d, maximum number of principal components possible: %d', num_principal_components, size(Sigma, 2));
end
% If the number of principal components requested is within the bounds
% return the first n columns of the U matrix.
principal_components = U * Sigma (:, 1:num_principal_components);
explained_var = sum(diag(Sigma(1:num_principal_components, 1:num_principal_components))) / sum(diag(Sigma));
end
