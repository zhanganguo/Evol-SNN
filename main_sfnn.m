%% Train an example FC network to achieve very high classification, fast.
%    Load paths
addpath(genpath('./dlt_cnn_map_dropout_nobiasnn'));
addpath(genpath('./models'));
addpath(genpath('./lifsim'));
rng('default')
 
%% Load dataset and model data
% dataset = 'fashion_mnist';
dataset = 'fashion_mnist';
if strcmp(dataset, 'fashion_mnist') == 1
    train_images = loadMNISTImages('train-images-idx3-ubyte');
    train_labels = loadMNISTLabels('train-labels-idx1-ubyte');
    test_images = loadMNISTImages('t10k-images-idx3-ubyte');
    test_labels = loadMNISTLabels('t10k-labels-idx1-ubyte');
    train_x = train_images';
    train_y = zeros(length(train_labels), 10);
    for i = 1 : length(train_labels)
        train_y(i, train_labels(i)+1) = 1;
    end
    test_x = test_images';
    test_y = zeros(length(test_labels), 10);
    for i = 1 : length(test_labels)
        test_y(i, test_labels(i)+1) = 1;
    end
    train_x = double(train_x);
    test_x  = double(test_x);
    train_y = double(train_y);
    test_y  = double(test_y);
    
    load nn_fashion_mnist_90.75.mat;
else
    load mnist_uint8;
    train_x = double(train_x) / 255;
    test_x  = double(test_x)  / 255;
    train_y = double(train_y);
    test_y  = double(test_y);
    
    load nn_mnist_98.84.mat;
end

%% Spike-based Testing of Fully-Connected NN
lifsim_opts = struct;
lifsim_opts.t_ref        = 0.000;
lifsim_opts.threshold    = 1.0;
lifsim_opts.rest         = 0.0;
lifsim_opts.dt           = 0.001;
lifsim_opts.duration     = 0.050;
lifsim_opts.report_every = 0.001;
lifsim_opts.max_rate     =   200;
sfnn = lifsim_sfnn(nn, test_x, test_y, lifsim_opts);

%% Options of Adaptive Rule 
evol_ops.beta = 0.6;
evol_ops.eta =0.5;
evol_ops.initial_E = 1;
evol_ops.learning_rate = 0.01;

evol_sfnn = lifsim_evol_sfnn(nn, test_x, test_y, lifsim_opts, evol_ops);
adap_evol_fnn = lifsim_adap_evol_sfnn(nn, test_x, test_y, lifsim_opts, evol_ops);

%% Show the difference
figure; clf;
time = lifsim_opts.dt:lifsim_opts.dt:lifsim_opts.duration;
plot(time * 1000, sfnn.performance);
hold on; grid on;
plot(time * 1000, evol_sfnn.performance);
hold on; grid on;
plot(time * 1000, adap_evol_fnn.performance);
legend('SFNN', 'Evol-SFNN','Adap-Evol-SFNN');
ylim([0 100]);
xlabel('Time [ms]');
ylabel('Accuracy [%]');
title('50Hz')