%% Homework 4 Part 1 
% This script is used to solve the Part 1 of the Homework 4 for 
% class CSCI 5521 - Introduction to Machine Learning. This part includes
% using the deltaNN function to train a neural network.

V0 = [.3 .3 .3; -.3 -.1 .1];
W0 = [ .20 .40 .10; -.16 -.44 -.09];
Data = [1 -1 1 -1; 1 1 -1 -1];
Labels = [-1 1 1 -1; 1 -1 -1 1];
rate=.1;

numEpochs = 100;
V = V0;
W = W0;

totalError = zeros(numEpochs);

for i = 1:numEpochs
    current_Error = 0;
    for sample_num = 1:size(Data, 2)
        x = Data(:, sample_num);
        t = Labels(:, sample_num);
        [dE_dV,dE_dW,E,z,y] = deltaNN(V,W,x,t);
        V = V - (rate * dE_dV);
        W = W - (rate * dE_dW);
        current_Error = current_Error + E;
    end
    totalError(i) = current_Error;
end

plot(totalError);