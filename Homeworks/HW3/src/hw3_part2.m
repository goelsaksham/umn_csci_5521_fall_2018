% This script is used to solve the Part 2 of Homework 3 for the class CSCI
% 5521 - Introduction to Machine Learning

% Part 2 asks us to apply PCA on the dataset to capture 90% of the variance
% and then apply K-Means Clustering on top of the resulting dataset and
% find the digits assigned to each cluster

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

%% Section 2: Applying K-Means Algorithm to get the Cluster Centers
number_Of_Cluster = 6;
initial_Cluster_Centers = X_projection([1; 1000; 1001; 2000; 2001; 3000], :);
%[clusters_Assigned_To_Points, clusterCenters] = my_kmeans(X_projection, number_Of_Cluster, initial_Cluster_Centers);
[clusters_Assigned_To_Points, clusterCenters, numIterations] = DoKmeans(X_projection, initial_Cluster_Centers);

%% Section 3: Finding the Confusion Matrix
% Need to make a matrix of 6 X 3 where we show the number of digits
% assigned to each cluster

%confusion_Matrix = zeros(6, 3);
%for cluster_Index = 1:number_Of_Cluster
%    points_Assigned_To_Cluster_True_Labels = y(clusters_Assigned_To_Points == cluster_Index);
%    % Find number of points assigned to current cluster for each digit
%    confusion_Matrix(cluster_Index, 1) = sum(points_Assigned_To_Cluster_True_Labels == 0);
%    confusion_Matrix(cluster_Index, 2) = sum(points_Assigned_To_Cluster_True_Labels == 8);
%    confusion_Matrix(cluster_Index, 3) = sum(points_Assigned_To_Cluster_True_Labels == 9);
%end
confusion_Matrix = GetConfusionMatrix(y, clusters_Assigned_To_Points);

%% Section 4: Finding the Error Rate
total_Number_Of_Misclassified_Digits = 0;
for cluster_Index = 1:number_Of_Cluster
    cluster_Classification = confusion_Matrix(cluster_Index, :);
    sorted_cluster_Classification = sort(cluster_Classification);
    total_Number_Of_Misclassified_Digits = total_Number_Of_Misclassified_Digits + sum(sorted_cluster_Classification(1:2));
end

error_Rate = total_Number_Of_Misclassified_Digits/size(y, 1);