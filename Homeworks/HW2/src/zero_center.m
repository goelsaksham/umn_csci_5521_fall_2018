function [zeroCenterFeatureMatrix] = zero_center(inputFeatureMatrix)
%ZERO_CENTER Summary of this function goes here
%   Detailed explanation goes here
zeroCenterFeatureMatrix = inputFeatureMatrix - mean(inputFeatureMatrix);
end

