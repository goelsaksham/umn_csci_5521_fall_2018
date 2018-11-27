function [y] = derivative_of_smooth_relu(x)

y = (exp(10*x) / (1 + exp(10*x)));

end

