function [assignments, centers, StepCount] = DoKmeans(data, InitialClusters)
%MY_K-MEANS Summary of this function goes here
%   Detailed explanation goes here

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
    % First do the E step, where each points is first initialized a cluster
    for pointIndex = 1:size(data, 1)
        pointCoordinates = data(pointIndex, :);
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

