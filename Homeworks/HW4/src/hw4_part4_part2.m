%% Homework 4 Part 2
% This script is used to solve the Part 2 of the Homework 4 for 
% class CSCI 5521 - Introduction to Machine Learning. This part includes
% using the deltaNN function to train a neural network.

%% Initializing the Parameters for the total RUN:
lr_s = [0.05, 0.1, 0.2];
num_epochs = [100, 200, 1600];

for lr_index = 1:size(lr_s, 2)
    for epoch_index = 1:size(num_epochs, 2)
        %% Initalizing the parameters
        rng('default');
        U0 = randn(2, 3);
        V0 = randn(3, 3);
        W0 = randn(2, 4);
        Data = [1 -1 1 -1; 1 1 -1 -1];
        Labels = [-1 1 1 -1; 1 -1 -1 1];
        rate=lr_s(lr_index);

        numEpochs = num_epochs(epoch_index);
        U = U0;
        V = V0;
        W = W0;

        totalError = zeros(numEpochs, 1);


        %% Training the Model using the Cyclic Gradient Descent Training Algo.
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

        %% Plotting the Training Error
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

        %% Analysis

        % Training Accuracy Error
        Total_Acc_Err_Num = 0;
        for sample_num = 1:size(Data, 2)
            x = Data(:, sample_num);
            t = Labels(:, sample_num);
            [~,~,~,~,z,~] = deltaNN_part_b(U, V,W,x,t);
            if ~is_correct_prediction(t, z)
                Total_Acc_Err_Num = Total_Acc_Err_Num + 1;
            end
        end
        Total_Acc_Err = Total_Acc_Err_Num / size(Data, 2);

        fprintf('k1: 2, k2: 3, Learning rate: %f, Num Epochs: %d, Last Training Error: %f, Accuracy Error: %f\n', rate, numEpochs, current_Error, Total_Acc_Err);
    end
end