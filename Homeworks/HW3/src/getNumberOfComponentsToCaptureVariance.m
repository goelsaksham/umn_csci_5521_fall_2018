function [numberOfComponents] = getNumberOfComponentsToCaptureVariance(explained_Variance, percentVariance)
%GETNUMBEROFCOMPONENTSTOSATISFYVARIANCE Summary of this function goes here
%   Detailed explanation goes here

for i = 1:size(explained_Variance, 1)
    if sum(explained_Variance(1:i, 1)) > percentVariance
        numberOfComponents = i;
        break
    end
end

end

