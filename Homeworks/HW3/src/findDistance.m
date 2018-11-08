function [distance] = findDistance(x1, x2)
%   FINDDISTANCE This function finds the euclidean distance between two
%   vectors.
%   
% Arguments:
%   x1: First Vector
%   x2: Second Vector

distance = sqrt(sum((x1 - x2).^2));

end

