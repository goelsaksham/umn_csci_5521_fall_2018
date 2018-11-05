function [imageFeatureMatrix, numRows, numCols] = getImageFeatureMatrix(imagePath)

imagePixelValues = imread(imagePath);
numRows = size(imagePixelValues, 1);
numCols = size(imagePixelValues, 2);
imageFeatureMatrix = reshape(imagePixelValues, [numRows * numCols, 3]);
end

