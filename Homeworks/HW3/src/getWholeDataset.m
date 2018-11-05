function [X,y] = getWholeDataset(fileName)
%GETWHOLEDATASET Summary of this function goes here
%   Detailed explanation goes here
all_Digits_Data = dlmread(fileName);
X = all_Digits_Data(:, 3:end);
y = all_Digits_Data(:, 2);
end

