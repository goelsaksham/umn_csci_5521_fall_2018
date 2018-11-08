function [numberOfComponents] = getNumberOfComponentsToCaptureVariance(explained_Variance, percentVariance)
%   GETNUMBEROFCOMPONENTSTOSATISFYVARIANCE returns the number of components
%   required to capture particular amount of variance. This function tries
%   to work on top of the PCA projections

% explained_Variance: The array denoting the amount of variance explained
% for each ith component.
% percentVariance: The target variance required to be captured.

for i = 1:size(explained_Variance, 1)
    if sum(explained_Variance(1:i, 1)) > percentVariance
        numberOfComponents = i;
        break
    end
end

end

