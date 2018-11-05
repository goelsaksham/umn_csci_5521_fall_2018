% This script is used to solve the Part 1 of Homework 3 for the class CSCI
% 5521 - Introduction to Machine Learning

% Part 1 asks us to apply PCA on the dataset to capture 90% of the variance

%% Section 1: Applying PCA on the whole dataset

% Load the whole dataset
[X, y] = getWholeDataset('./data/Digits089.csv');
% Zero center the data
X = X - mean(X);
% Apply PCA
[projection_Vectors, ~, ~, ~, explained_Variance] = pca(X);
% Get number of components for capturing 90% variance
numComponents = getNumberOfComponentsToCaptureVariance(explained_Variance, 90.0);
% Get the projections of the data onto the number of components
X_projection = getSubspaceProjections(numComponents, projection_Vectors, X);

%% Section 2: Applying PCA on training data and projecting test data on top of those vectors

% Do we need to do this??