function [y] = smooth_relu(x)
% This function is used to compute the smooth Relu value. This derivative
% is expecting the input to be the actual input of the smooth relu
% function.

y = (1/10) * log(1 + exp(10 * x));

end

