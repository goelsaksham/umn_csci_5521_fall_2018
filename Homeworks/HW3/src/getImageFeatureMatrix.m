function [imageFeatureMatrix, numRows, numCols] = getImageFeatureMatrix(imagePath)
% This function is used to get the Feature Matrix from the image. This
% function basically converts the 3 Dimensional Image Channels Matrix to a
% 2 Dimensional sample matrix such that each row corresponds to the RGB
% channel of each particular pixel. Also returns the height and width of
% the image.
%
% Arguments:
%   imagePath: The path to the image which should be loaded
%
imagePixelValues = imread(imagePath);
numRows = size(imagePixelValues, 1);
numCols = size(imagePixelValues, 2);
imageFeatureMatrix = reshape(imagePixelValues, [numRows * numCols, 3]);
end

