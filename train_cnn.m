%% Train an example ConvNet to achieve very high classification, fast.
dbstop if error;
addpath(genpath('./dlt_cnn_map_dropout_nobiasnn'));
addpath(genpath('./models'));
rng('default')

%% Load data
load mnist_uint8;
train_x = double(reshape(train_x',28,28,60000)) / 255;
test_x = double(reshape(test_x',28,28,10000)) / 255;
train_y = double(train_y');
test_y = double(test_y');

%% Setup CNN
cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 16, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
    struct('type', 'c', 'outputmaps', 64, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %subsampling layer
};

cnn = cnnsetup(cnn, train_x, train_y);
% Set the activation function to be a ReLU
cnn.act_fun = @(inp)max(0, inp);
% Set the derivative to be the binary derivative of a ReLU
cnn.d_act_fun = @(forward_act)double(forward_act>0);

%% ReLU Train
% Set up learning constants
opts.alpha = 1;
opts.batchsize = 50;
opts.numepochs =  20;
opts.learn_bias = 0;
opts.dropout = 0.5;
cnn.first_layer_dropout = 0;
% Train
cnn = cnntrain(cnn, train_x, train_y, opts);
% Test
[er, bad] = cnntest(cnn, train_x, train_y);
fprintf('TRAINING Accuracy: %2.2f%%.\n', (1-er)*100);
[er, bad] = cnntest(cnn, test_x, test_y);
fprintf('Test Accuracy: %2.2f%%.\n', (1-er)*100);

%% Save the trained model data
save cnn