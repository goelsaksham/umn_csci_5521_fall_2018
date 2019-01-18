function [label] = label_problem2_partb(x, y)
% Returns the label as 1 if the vector [x; y] is in the shaded region as
% seen in the Question 2 Part A
% Returns a label based on whether the vector lies in the shaded region.
% The return label is 1 if vector lies in the shaded region, else 0
label = 0;

if (y >= -1.5) && ((x - (2*y) - 2) > 0)
    label = 1;
end

end