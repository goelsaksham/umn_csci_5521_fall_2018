function [normalized_data] = normalize(input_data)

% Estimate the mean and the standard deviation of the 
input_data_mean = mean(input_data);
input_data_std = std(input_data);

normalized_data = (input_data - input_data_mean)./input_data_std;


end

