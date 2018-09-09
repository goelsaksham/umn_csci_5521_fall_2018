function [w, steps] = MyPerceptron(X, y, w0)
% MyPerceptron runs the basic Perceptron algorithm discussed in the given
% Hwk-0 pdf.
%
% Args:
%   X - A matrix representing the training data
%   y - A vector of 1's or -1's representing the class to which the data
%       belongs
%   w0 - The vector of initial Value of the parameters that need to be 
%        learned to classify the data
%
% Returns:
%   w - The vector representing the learned parameters that can classify
%       the training data as best as possible after runnning the perceptron
%       algorithm.
%   steps - The number of iterations it took for the perceptron algorithm
%           to converge


% Plot the data before running the perceptron algorithm.
figure(1);
scatter(X(:, 1), X(:, 2), 50, y, '*');
hold on;
title('initialization');
xlim('manual');
ylim('manual');
% Plot the Initial Line denoted by w0 (initial parameters)
plot([1, -1], [(-(w0(1))/(w0(2))), ((w0(1))/(w0(2)))]);
hold off;

xlim('auto');
ylim('auto');

% Find the number of training examples
n = size(X, 1);

% Initialize variables
w = w0;
prev_w = w0 + ones(size(w));
steps = 0;

% Run the perceptron algorithm
while ~isequal(w, prev_w)
    steps = steps + 1;
    prev_w = w;
    for i = 1:n
        if y(i) * (X(i, :) * w) <= 0
            w = w + y(i) .* transpose(X(i, :));
        end
    end
end

% Plot the data after running the perceptron algorithm.
figure(2);
scatter(X(:, 1), X(:, 2), 50, y, '*');
hold on;
title('when perceptron converge');
xlim('manual');
ylim('manual');
% Plot the Initial Line denoted by w0 (initial parameters)
plot([1, -1], [(-(w(1))/(w(2))), ((w(1))/(w(2)))]);
hold off;