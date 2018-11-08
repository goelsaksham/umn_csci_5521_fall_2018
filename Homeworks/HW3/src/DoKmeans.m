function [assignments, centers, StepCount] = DoKmeans(data, InitialClusters)
%   DoKmeans is my implementation of the K-Means algorithm. This function
%   basically runs the K-Means algorithm on the given dataset and returns
%   the assignments for each data point along with the found cluster
%   centers and the number of steps it took to converge.

% Parameters
%   data: A M X N matrix containing the M number of data samples all with N
%   features (in N Dimensional Space)
%   InitialClusters: A matrix containg the initial centers for the K-Means
%   Cluster centers. Number of rows in this matrix determine the number of
%   Clusters. Each cluster center corresponds to the row vector.
%
% Define the vector which represents the cluster to which each point has been
% assigned
clusterLabels = zeros(size(data, 1), 1);
% Get the number of clusters
numberOfClusters = size(InitialClusters, 1);
% Initialize the new cluster centers
newClusterCenters = InitialClusters;
% Initialize the stepcount
StepCount = 0;

while 1
    saveClusterCenters = newClusterCenters;
    % First do the E step, where each points is first assigned to a cluster
    % For this part, we basically just iterate through each point and find
    % the closest cluster and assign it to that
    for pointIndex = 1:size(data, 1)
        pointCoordinates = data(pointIndex, :);
        minDistance = inf;
        pointClusterLabel = 0;
        % To find the closest cluster we loop through all the clusters
        for clusterIndex = 1:numberOfClusters
            clusterCoordinates = newClusterCenters(clusterIndex, :);
            distanceBtwClusterAndPoint = findDistance(pointCoordinates, clusterCoordinates);
            if distanceBtwClusterAndPoint < minDistance
                minDistance = distanceBtwClusterAndPoint;
                pointClusterLabel = clusterIndex;
            end
        end
        clusterLabels(pointIndex) = pointClusterLabel;
    end
    
    % Then do the M step, where each cluster is recomputed by calculating
    % the mean of all the points assigned to that cluster and this becomes
    % the new initialization of the cluster center
    for clusterIndex = 1:numberOfClusters
        pointsAssignedToCluster = data(clusterLabels == clusterIndex, :);
        newClusterCenter = mean(pointsAssignedToCluster);
        newClusterCenters(clusterIndex, :) = newClusterCenter;
    end
    StepCount = StepCount + 1;
    % Check if the algorithm has converged
    if newClusterCenters == saveClusterCenters
        break
    end
end

assignments = clusterLabels;
centers = newClusterCenters;
end

