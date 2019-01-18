function [out] = sigmoid(x)
% Sigmoid Activation Function
% x -> input
% out -> Sigmoid Function Activation Value
out = 1 / (1 + exp(-x));
end

