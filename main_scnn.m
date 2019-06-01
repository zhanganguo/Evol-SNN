%% Train an example ConvNet to achieve very high classification, fast.
dbstop if error;
addpath(genpath('./dlt_cnn_map_dropout_nobiasnn'));
addpath(genpath('./models'));
addpath(genpath('./util'));
rng('default')

%% Load data and network model
% dataset = 'fashion_mnist';
dataset = 'mnist';
if strcmp(dataset, 'fashion_mnist') == 1
    train_images = loadMNISTImages('train-images-idx3-ubyte');
    train_labels = loadMNISTLabels('train-labels-idx1-ubyte');
    test_images = loadMNISTImages('t10k-images-idx3-ubyte');
    test_labels = loadMNISTLabels('t10k-labels-idx1-ubyte');
    train_y = zeros(length(train_labels), 10);
    for i = 1 : length(train_labels)
        train_y(i, train_labels(i)+1) = 1;
    end
    test_y = zeros(length(test_labels), 10);
    for i = 1 : length(test_labels)
        test_y(i, test_labels(i)+1) = 1;
    end
    train_x = double(reshape(train_images,28,28,60000));
    test_x = double(reshape(test_images,28,28,10000));
    train_y = double(train_y');
    test_y = double(test_y');
    % Load a trained ANN model
    load cnn_fashion_mnist_91.35.mat;
elseif strcmp(datasete, 'mnist') == 1
    load mnist_uint8;
    train_x = double(reshape(train_x',28,28,60000)) / 255;
    test_x = double(reshape(test_x',28,28,10000)) / 255;
    train_y = double(train_y');
    test_y = double(test_y');
    % Load a trained ANN model
    load cnn_99.14.mat;
end
    
%% Spike-based Testing of a ConvNet
lifsim_opts = struct;
lifsim_opts.t_ref        = 0.000;
lifsim_opts.threshold    =   1.0;
lifsim_opts.dt           = 0.001;
lifsim_opts.duration     = 0.100;
lifsim_opts.report_every = 0.001;
lifsim_opts.max_rate     =  200;

%% Test the original SCNN
scnn = lifsim_scnn(cnn, test_x, test_y, lifsim_opts);

%% Test the Evolutionary Rule 
evol_ops.beta = 0.6;
evol_ops.eta =0.5;
evol_ops.initial_Ec = 1;
evol_ops.learning_rate = 0.01;

evol_scnn = lifsim_evol_scnn(cnn, test_x, test_y, lifsim_opts, evol_ops);
adap_evol_scnn = lifsim_adap_evol_scnn(cnn, test_x, test_y, lifsim_opts, evol_ops);

% plot_cnn_spikes(scnn);
% plot_cnn_spikes(evol_scnn);
% plot_cnn_spikes(adap_evol_scnn);

%% Show the difference
figure; clf;
time = lifsim_opts.dt:lifsim_opts.dt:lifsim_opts.duration;
plot(time * 1000, scnn.performance);
hold on; grid on;
plot(time * 1000, evol_scnn.performance);
hold on; grid on;
plot(time * 1000, adap_evol_scnn.performance);
legend('SCNN', 'Evol-SCNN', 'Adap-Evol-SCNN');
ylim([00 100]);
xlabel('Time [ms]');
ylabel('Accuracy [%]');