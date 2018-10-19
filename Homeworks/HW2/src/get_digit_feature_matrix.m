function [digitFeatureMatrix, digitVector] = get_digit_feature_matrix(inputfeatureMatrix, targetValuesVector, digit)
%GET_DIGIT_DATA Summary of this function goes here
%   Detailed explanation goes here
digitFeatureMatrix = inputfeatureMatrix(targetValuesVector == digit, :);
digitVector = targetValuesVector(targetValuesVector == digit, 1);
end

