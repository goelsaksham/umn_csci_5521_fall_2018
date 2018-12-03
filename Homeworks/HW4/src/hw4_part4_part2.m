%% Homework 4 Part 1 
% This script is used to solve the Part 1 of the Homework 4 for 
% class CSCI 5521 - Introduction to Machine Learning. This part includes
% using the deltaNN function to train a neural network.

U0 = rand(2, 3);
V0 = rand(3, 3);
W0 = rand(2, 4);
Data = [1 -1 1 -1; 1 1 -1 -1];
Labels = [-1 1 1 -1; 1 -1 -1 1];
rate=.1;

numEpochs = 100;
U = U0;
V = V0;
W = W0;

totalError = zeros(numEpochs, 1);

for i = 1:numEpochs
    current_Error = 0;
    for sample_num = 1:size(Data, 2)
        x = Data(:, sample_num);
        t = Labels(:, sample_num);
        [dE_dU,dE_dV,dE_dW,E,z,y] = deltaNN_part_b(U, V,W,x,t);
        U = U - (rate * dE_dU);
        V = V - (rate * dE_dV);
        W = W - (rate * dE_dW);
        current_Error = current_Error + E;
    end
    totalError(i) = current_Error;
end

plot(totalError);
disp(totalError);