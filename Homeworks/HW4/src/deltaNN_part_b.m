function [dE_dK,dE_dV,dE_dW,E,z,y]=deltaNN_part_b(U,V,W,u,t);
% function [dE_dV,dE_dW,E,z,y]=deltaNN(V,W,x,t);
% compute derivative of square-norm Error funcional
% wrt V,W coeffs in a neural network with one hidden layer:
%  u -> U -> x -> V -> tanh -> y -> W -> tanh -> z
%  x = input and is M x 1, V is N x M+1, W is P x N+1, 
% The input parameters are:
% - U (M x K + 1): Weights between each input unit and each hidden unit 
% - V (N x M + 1): Weights between each hidden unit and each hidden unit 
% - W (P x N + 1): Weights between each hidden unit and each output unit
% - u (K x 1): input data, M is the number of features in one input datum
% - t (P x 1): True labels for the given datum
% The function should return:
% - dE_dU (M x K+1)
% - dE_dV (N x M+1): Updates for the weights in V
% - dE_dW (P x N+1): Updates for the weights in W
% - E (1 x 1): The error (a scalar)
% - z (P x 1): Output from each output unit
% - x (M x 1): Output from each hidden unit
% - y (N x 1): Output from each hidden unit

% get dimensions
[M,Kp1] = size(U);
[N,Mp1]=size(V);
[P,o]=size(W);
K = Kp1 - 1;




% Apply inputs to network to compute node values
xhat=zeros(M,1); for j=1:M;xhat(j)=U(j,:)*[1;u]; end
x=zeros(M,1);for j=1:M;x(j)=tanh(xhat(j));end

yhat=zeros(N,1); for j=1:N;yhat(j)=V(j,:)*[1;x]; end
y=zeros(N,1);for j=1:N;y(j)=tanh(yhat(j));end
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
  for j=1:N; 
     acc=0; for l=1:P; acc=acc+dE_dzhat(l)*W(l,1+j); end;
     dE_dyhat(j)=acc*(1-y(j)^2);
  end;

% (d) partial derivative of E with respect to V:
dE_dV=zeros(N,M+1);
  for j=1:N;
    dE_dV(j,1+0)=dE_dyhat(j);
    for k=1:M; dE_dV(j,1+k)=dE_dyhat(j)*x(k);end;
  end;
 
 %% Updating the U Weights 
  % (b) gamma_j = dE_dyhat(j):
dE_dxhat=zeros(M,1);
  for j=1:M; 
     acc=0; for l=1:N; acc=acc+dE_dyhat(l)*V(l,1+j); end;
     dE_dxhat(j)=acc*(1-x(j)^2);
  end;

% (d) partial derivative of E with respect to V:
dE_dK=zeros(M,K+1);
  for j=1:M;
    dE_dK(j,1+0)=dE_dxhat(j);
    for k=1:K; dE_dK(j,1+k)=dE_dxhat(j)*u(k);end;
  end;
