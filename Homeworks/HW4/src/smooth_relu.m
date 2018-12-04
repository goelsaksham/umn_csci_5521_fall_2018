function [y] = smooth_relu(x)
% This function computes the smooth relu actiavation function value.
y = (1/10) * log(1 + exp(10 * x));
end

