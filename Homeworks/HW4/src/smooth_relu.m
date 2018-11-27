function [y] = smooth_relu(x)

y = (1/10) * log(1 + exp(10 * x));

end

