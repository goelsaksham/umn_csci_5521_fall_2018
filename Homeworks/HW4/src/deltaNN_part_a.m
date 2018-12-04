function [dE_dV,dE_dW,E,z,y]=deltaNN_part_a(V,W,x,t);
% function [dE_dV,dE_dW,E,z,y]=deltaNN(V,W,x,t);
% compute derivative of square-norm Error funcional
% wrt V,W coeffs in a neural network with one hidden layer:
%  x ->V->tanh-> y ->W->tanh-> z
%  x = input and is M x 1, V is N x M+1, W is P x N+1, 
% The input parameters are:
% - V (M x N+1): Weights between each input unit and each hidden unit 
% - W (N x P+1): Weights between each hidden unit and each output unit
% - x (M x 1):input data, M is the number of features in one input datum
% - t (P x 1): True labels for the given datum
% The function should return:
% - dE_dV (M x N+1): Updates for the weights in V
% - dE_dW (N x P+1): Updates for the weights in W
% - E (1 x 1): The error (a scalar)
% - z (P x 1): Output from each output unit
% - y (N x 1): Output from each hidden unit

% get dimensions
[N,Mp1]=size(V);
[P,o]=size(W);
M=Mp1-1;

% Apply inputs to network to compute node values
yhat=zeros(N,1); for j=1:N;yhat(j)=V(j,:)*[1;x]; end
y=zeros(N,1);for j=1:N;y(j)=smooth_relu(yhat(j));end
zhat=zeros(P,1);for i=1:P;zhat(i)=W(i,:)*[1;y]; end
z=zeros(P,1);for i=1:P;z(i)=tanh(zhat(i));end;
% Compute error: discrepancy between computed outputs and true targets:
E=.5*(norm(z-t))^2;

% Compute derivatives of error functional as described in handout.
% (a) delta_i = dE_dzhat(i):
dE_dzhat=zeros(P,1);for i=1:P; dE_dzhat(i)=(z(i)-t(i))*(1-z(i)^2); end;

% (c) partial derivative of E with respect to W:
dE_dW=zeros(P,N+1);
  for i=1:P;
    dE_dW(i,1+0)=dE_dzhat(i);
    for j=1:N; dE_dW(i,1+j)=dE_dzhat(i)*y(j);end;
  end;

% (b) gamma_j = dE_dyhat(j):
dE_dyhat=zeros(N,1);
  for j=1:N
     acc=0; 
     for l=1:P 
         acc=acc+dE_dzhat(l)*W(l,1+j); 
     end
     dE_dyhat(j)=acc*derivative_of_smooth_relu(y(j));
  end

% (d) partial derivative of E with respect to V:
dE_dV=zeros(N,M+1);
  for j=1:N;
    dE_dV(j,1+0)=dE_dyhat(j);
    for k=1:M; dE_dV(j,1+k)=dE_dyhat(j)*x(k);end;
  end;
 
