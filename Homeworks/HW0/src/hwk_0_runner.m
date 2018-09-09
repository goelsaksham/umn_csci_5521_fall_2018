% ################################################
%
% This script is used to run the various commands
% associated with the Homework 0 of UMN CSCI 5521
% Intro to Machine Learning Class.
%
% ################################################


% Load the Data from ./data/data1.mat file
data_1 = load('./data/data1.mat');

% Get the X (Input) and y (Output) Matrices/Vectors.
X = data_1.X;
y = data_1.y;

% Get the number of training examples
n = size(X, 1);

% Get the number of parameters
m = size(X, 2);

[w, steps] = MyPerceptron(X, y, [1; -1]);


pause;


% Load the Data from ./data/data2.mat file
data_2 = load('./data/data2.mat');

% Get the X (Input) and y (Output) Matrices/Vectors.
X = data_2.X;
y = data_2.y;

% Get the number of training examples
n = size(X, 1);

% Get the number of parameters
m = size(X, 2);

[w, steps] = MyPerceptron(X, y, [1; -1]);