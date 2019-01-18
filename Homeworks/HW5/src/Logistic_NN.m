function [dE_dV,E,z]=One_Layer_NN(V,x,t,activation_function,derivative_of_activation_function)
% function [dE_dV,E,z]=One_Layer_NN(V,x,t);
% compute derivative of square-norm Error funcional
% wrt V coeffs in a neural network with no hidden layer:
%  x -> V -> sigmoid -> z
%  x = input and is M x 1, V is 1 x M+1
% The input parameters are:
% - V (1 x M+1): Weights between each input unit and the output unit 
% - x (M x 1):input data, M is the number of features in one input datum
% - t (1 x 1): True label for the given datum
% The function should return:
% - dE_dV (1 x M+1): Updates for the weights in V
% - E (1 x 1): The error (a scalar)
% - z (1 x 1): Output from output unit

% get dimensions
[N , ~] = size(V);

% Apply inputs to network to compute node values
zhat = V * [1; x];
z = activation_function(zhat);
% size(V)
% size(zhat)
% size(z)

% Compute error: discrepancy between computed outputs and true targets:
E = ((norm(z - t)) ^ 2) / 2;

% Compute derivatives of error functional as described in handout.
% (a) delta_i = dE_dzhat(i):
dE_dzhat = (z - t) * derivative_of_activation_function(zhat);

% (c) partial derivative of E with respect to W:

dE_dV = dE_dzhat .* [1; x];
dE_dV = transpose(dE_dV);