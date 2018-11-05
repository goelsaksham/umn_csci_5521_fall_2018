function [clusterAssignments, clusterCenters] = my_kmeans(featureMatrix, numberOfClusters, clusterInitialCenters)
%MY_K-MEANS Summary of this function goes here
%   Detailed explanation goes here

% Define the vector which represents the cluster to which each point has been
% assigned
clusterLabels = zeros(size(featureMatrix, 1), 1);
% Initialize the new cluster centers
newClusterCenters = clusterInitialCenters;

while 1
    saveClusterCenters = newClusterCenters;
    % First do the E step, where each points is first initialized a cluster
    for pointIndex = 1:size(featureMatrix, 1)
        pointCoordinates = featureMatrix(pointIndex, :);
        minDistance = inf;
        pointClusterLabel = 0;
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
    
    % Then do the M step, where each cluster is recomputed
    for clusterIndex = 1:numberOfClusters
        pointsAssignedToCluster = featureMatrix(clusterLabels == clusterIndex, :);
        newClusterCenter = mean(pointsAssignedToCluster);
        newClusterCenters(clusterIndex, :) = newClusterCenter;
    end
    
    % Check if the algorithm has converged
    if newClusterCenters == saveClusterCenters
        break
    end
end

clusterAssignments = clusterLabels;
clusterCenters = newClusterCenters;
end

