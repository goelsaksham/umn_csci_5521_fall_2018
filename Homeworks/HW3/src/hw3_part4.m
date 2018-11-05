% This script is used to solve the Part 4 of Homework 3 for the class CSCI
% 5521 - Introduction to Machine Learning

% Part 4 asks us to apply K-Means Clustering on the images and redraw them
% based on the new cluster centers

%% Section 1: Get Image Data for goldy.ppm
Clusters_Numbers = [3; 4; 7];
for cluster_Index = 1:size(Clusters_Numbers, 1)
    % Get the image feature matrix
    [image_Feature_Matrix, numRows, numCols] = getImageFeatureMatrix('./data/goldy.ppm');
    
    figure(4);
    imagesc(reshape(reshape(image_Feature_Matrix, [numRows*numCols*3, 1]), [numRows numCols 3]));
    
    image_Feature_Matrix = int32(image_Feature_Matrix);
    
    number_of_cluster = Clusters_Numbers(cluster_Index);
    cluster_Centers = rand(number_of_cluster, 3) * 255;
    cluster_Centers = int32(cluster_Centers);
    % Apply the k-means algorithm with different number of cluster centers
    %[cluster_Labels, cluster_Centers] = my_kmeans(image_Feature_Matrix, number_of_cluster, cluster_Centers);
    [cluster_Labels, cluster_Centers, numIters] = DoKmeans(image_Feature_Matrix, cluster_Centers);
    for feature_Index = 1:size(image_Feature_Matrix, 1)
        image_Feature_Matrix(feature_Index, :) = cluster_Centers(cluster_Labels(feature_Index), :);
    end
    reshapedImage = reshape(reshape(image_Feature_Matrix, [numRows*numCols*3, 1]), [numRows numCols 3]);
    reshapedImage = uint8(reshapedImage);
    figure(cluster_Index);
    imagesc(reshapedImage);
end


%% Section 2: Get Image Data for stadium.ppm
Clusters_Numbers = [3; 4; 7];
for cluster_Index = 1:size(Clusters_Numbers, 1)
    % Get the image feature matrix
    [image_Feature_Matrix, numRows, numCols] = getImageFeatureMatrix('./data/stadium.ppm');
    
    figure(8);
    imagesc(reshape(reshape(image_Feature_Matrix, [numRows*numCols*3, 1]), [numRows numCols 3]));
    
    image_Feature_Matrix = int32(image_Feature_Matrix);
    
    number_of_cluster = Clusters_Numbers(cluster_Index);
    cluster_Centers = rand(number_of_cluster, 3) * 255;
    cluster_Centers = int32(cluster_Centers);
    % Apply the k-means algorithm with different number of cluster centers
    %[cluster_Labels, cluster_Centers] = my_kmeans(image_Feature_Matrix, number_of_cluster, cluster_Centers);
    [cluster_Labels, cluster_Centers, numIters] = DoKmeans(image_Feature_Matrix, cluster_Centers);
    for feature_Index = 1:size(image_Feature_Matrix, 1)
        image_Feature_Matrix(feature_Index, :) = cluster_Centers(cluster_Labels(feature_Index), :);
    end
    reshapedImage = reshape(reshape(image_Feature_Matrix, [numRows*numCols*3, 1]), [numRows numCols 3]);
    reshapedImage = uint8(reshapedImage);
    figure(4+cluster_Index);
    imagesc(reshapedImage);
end