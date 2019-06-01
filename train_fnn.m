dbstop if error
%% Train an example FC network to achieve very high classification, fast.
%    Load paths
addpath(genpath('./dlt_cnn_map_dropout_nobiasnn'));
%% Load data
rand('state', 0);
load mnist_uint8;

train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);

%% Initialize net
nn = nnsetup([784 1200 1200 10]);
% Rescale weights for ReLU
for i = 2 : nn.n   
    % Weights - choose between [-0.1 0.1]
    nn.W{i - 1} = (rand(nn.size(i), nn.size(i - 1)) - 0.5) * 0.01 * 2;
    nn.vW{i - 1} = zeros(size(nn.W{i-1}));
end
% Set up learning constants
nn.activation_function = 'relu';
nn.output ='relu';
nn.learningRate = 1;
nn.momentum = 0.5;
nn.dropoutFraction = 0.5;
nn.learn_bias = 0;
opts.numepochs =  200;
opts.batchsize = 100;
% Train - takes about 15 seconds per epoch on my machine
nn = nntrain(nn, noisy_train_x, train_y, opts);
% Test - should be 98.62% after 15 epochs
[er, train_bad] = nntest(nn, noisy_train_x, train_y);
fprintf('TRAINING Accuracy: %2.2f%%.\n', (1-er)*100);
[er, bad] = nntest(nn, noisy_test_x, test_y);
fprintf('Test Accuracy: %2.2f%%.\n', (1-er)*100);

%% Save the trained model data
save nn