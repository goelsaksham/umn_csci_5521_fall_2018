function [classPrediction] = kNN(K, testFeatureVector, trainingDataFeatureMatrix,trainingDataTargetVector)
%KNN Summary of this function goes here
%   Detailed explanation goes here

% First finding the raw distance between the testfeature vector and all the
% training samples
distances = sum((trainingDataFeatureMatrix - testFeatureVector).^2, 2);
[sortedDistance, indices] = sort(distances);
sortedDistanceClassLabels = trainingDataTargetVector(indices);
nearestKPointsClassLabels = sortedDistanceClassLabels(1:K);

allClassLabels = unique(trainingDataTargetVector);

difference = abs(sum(nearestKPointsClassLabels == allClassLabels(1))- sum(nearestKPointsClassLabels == allClassLabels(2)));

if (difference == 1) && (sortedDistance(K) == sortedDistance(K+1)) && (sortedDistanceClassLabels(K) ~= sortedDistanceClassLabels(K+1))
        classPrediction = -1;
else
    if sum(nearestKPointsClassLabels == allClassLabels(1)) > sum(nearestKPointsClassLabels == allClassLabels(2))
        classPrediction = allClassLabels(1);
    else
        classPrediction = allClassLabels(2);
    end
end
end

