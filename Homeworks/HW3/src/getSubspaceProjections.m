function [subspaceProjection] = getSubspaceProjections(numberOfDimensions, projectionVectors, featureMatrix)
%   GETSUBSPACEPROJECTIONS returns the projection of the samples on top of
%   the new projection vectors obtained from the Principal Component
%   Analysis
subspaceVectors = projectionVectors(:, 1:numberOfDimensions);
subspaceProjection = featureMatrix * subspaceVectors;
end

