function [classPrediction] = kNN(K, testFeatureVector, trainingDataFeatureMatrix,trainingDataTargetVector)
%KNN Summary of this function goes here
%   Detailed explanation goes here

% First finding the raw distance between the testfeature vector and all the
% training samples
allClassLabels = unique(trainingDataTargetVector);
distances = sum((trainingDataFeatureMatrix - testFeatureVector).^2, 2);
[sortedDistance, indices] = sort(distances);
sortedDistanceClassLabels = trainingDataTargetVector(indices);

end_index = K;
while sortedDistance(K) == sortedDistance(end_index+1)
    end_index = end_index + 1;
end
   
nearestKPointsClassLabels = sortedDistanceClassLabels(1:end_index);

if sum(nearestKPointsClassLabels == allClassLabels(1)) > sum(nearestKPointsClassLabels == allClassLabels(2))
   classPrediction = allClassLabels(1);
elseif sum(nearestKPointsClassLabels == allClassLabels(1)) < sum(nearestKPointsClassLabels == allClassLabels(2))
   classPrediction = allClassLabels(2);
else
    classPrediction = -1;
end
end

