%% Homework 4 Part 2
% This script is used to solve the Part 2 of the Homework 4 for 
% class CSCI 5521 - Introduction to Machine Learning. This part includes
% using the deltaNN function to train a neural network.

load('data/optdigits.txt');
all_labels=optdigits(:,65);
Data = [optdigits(all_labels==9 | all_labels==8,1:64)/256]';
Labels = [optdigits((all_labels==9 | all_labels==8),65)]';
Labels(Labels==9)=-1;
Labels(Labels==8)= 1;
initVW; % only for the first run.
rate=.1;

new_Labels = zeros(2, size(Labels, 2));

new_Labels(1, Labels == 1) = 1;
new_Labels(2, Labels == 1) = -1;

new_Labels(1, Labels == -1) = -1;
new_Labels(2, Labels == -1) = 1;

numEpochs = 400;
V = V0;
W = W0;

totalError = zeros(numEpochs, 1);

for i = 1:numEpochs
    current_Error = 0;
    for sample_num = 1:size(Data, 2)
        x = Data(:, sample_num);
        t = new_Labels(:, sample_num);
        [dE_dV,dE_dW,E,z,y] = deltaNN(V,W,x,t);
        V = V - (rate * dE_dV);
        W = W - (rate * dE_dW);
        current_Error = current_Error + E;
    end
    totalError(i) = current_Error;
end

figure(1);
plot(totalError);
title('Error v/s Epoch #');
xlabel('Epoch #');
ylabel('Training Error');
figure(2);
semilogy(totalError);
title('Error v/s Epoch #');
xlabel('Epoch #');
ylabel('Training Error');