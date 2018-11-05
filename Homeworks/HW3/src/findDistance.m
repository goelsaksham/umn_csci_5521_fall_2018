function [distance] = findDistance(x1, x2)
%FINDDISTANCE Summary of this function goes here
%   Detailed explanation goes here

distance = sqrt(sum((x1 - x2).^2));

end

