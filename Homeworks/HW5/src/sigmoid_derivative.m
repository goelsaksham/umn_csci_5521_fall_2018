function [der] = sigmoid_derivative(x)
% Derivative of the sigmoid function
% x -> Input value
% der -> Output Value representing the derivative
der = sigmoid(x) .* (1 - sigmoid(x));
end

