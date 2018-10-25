function [classPrediction] = kNN(K, testFeatureVector, trainingDataFeatureMatrix,trainingDataTargetVector)
%KNN Summary of this function goes here
%   Detailed explanation goes here

% First finding the raw distance between the testfeature vector and all the
% training samples
distances = sum((trainingDataFeatureMatrix - testFeatureVector).^2, 2);
[sorted_distances, indices] = sort(distances);
sortedDistanceClassLabels = trainingDataTargetVector(indices);
nearestKPointsClassLabels = sortedDistanceClassLabels(1:K);
nearestEvenPointsClassLabels = sortedDistanceClassLabels(1:K+1);
even_distances = sorted_distances(K+1);

allClassLabels = unique(trainingDataTargetVector);

if unique(even_distances) == sorted_distances(1)
    if sum(nearestEvenPointsClassLabels == allClassLabels(1)) == sum(nearestEvenPointsClassLabels == allClassLabels(2))
        classPrediction = -1;
    end
elseif sum(nearestKPointsClassLabels == allClassLabels(1)) > sum(nearestKPointsClassLabels == allClassLabels(2))
    classPrediction = allClassLabels(1);
else
    classPrediction = allClassLabels(2);
end

end

