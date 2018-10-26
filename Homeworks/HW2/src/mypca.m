function [projection_on_principal_components, explained_var] = mypca(feature_matrix, num_principal_components)
% Our function that implements the PCA algorithm to reduce the
% dimensionality of the given input data
% feature_matrix - Represents a M X N matrix containing M examples with N
% features each for the input data.
% num_principal_components - The number of dimensions on which the data
% should be projected

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
projection_on_principal_components = U * Sigma (:, 1:num_principal_components);

% If want the data not to be scaled
%projection_on_principal_components = U(:, 1:num_principal_components);

% Calculate the explained variance using the Eigen Values.
diag_Sigma = diag(Sigma);
%explained_var = sum(diag(Sigma(1:num_principal_components, 1:num_principal_components) .^ 2)) / sum(diag(Sigma .^ 2));
explained_var = sum(diag_Sigma(1:num_principal_components) .^ 2) / sum(diag_Sigma.^ 2);

% If want the principal component vectors:
%V = Vt';
% principal_component_vectors = V(:, 1:num_principal_components);

end
