function [X,y] = getWholeDataset(fileName)
% GETWHOLEDATASET Function that returns the whole dataset
% 
% Arguments:
%   fileName - The path to the csv file which holds the data for the digits
all_Digits_Data = dlmread(fileName);
X = all_Digits_Data(:, 3:end);
y = all_Digits_Data(:, 2);
end

