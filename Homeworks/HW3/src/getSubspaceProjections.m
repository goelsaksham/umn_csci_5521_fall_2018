function [subspaceProjection] = getSubspaceProjections(numberOfDimensions, projectionVectors, featureMatrix)
%GETSUBSPACEPROJECTIONS Summary of this function goes here
%   Detailed explanation goes here
subspaceVectors = projectionVectors(:, 1:numberOfDimensions);
subspaceProjection = featureMatrix * subspaceVectors;
end

