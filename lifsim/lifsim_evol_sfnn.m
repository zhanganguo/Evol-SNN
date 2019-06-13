function nn=lifsim_evol_sfnn(nn, test_x, test_y, lifsim_opts, evol_opts)
dt = lifsim_opts.dt;
nn.performance = [];
num_examples = size(test_x,1);

beta = evol_opts.beta;
initial_E = evol_opts.initial_E;
learning_rate = evol_opts.learning_rate;

% Initialize network architecture
for l = 1 : numel(nn.size)
    blank_neurons = zeros(num_examples, nn.size(l));
    one_neurons = ones(num_examples, nn.size(l));
    nn.layers{l}.mem = blank_neurons;
    nn.layers{l}.refrac_end = blank_neurons;
    nn.layers{l}.sum_spikes = blank_neurons;

    nn.layers{l}.E = one_neurons * initial_E;
end

% Precache answers
[~,   ans_idx] = max(test_y');

% Time-stepped simulation
for t=dt:dt:lifsim_opts.duration
    % Create poisson distributed spikes from the input images
    %   (for all images in parallel)
    rescale_fac = 1/(dt*lifsim_opts.max_rate);
    spike_snapshot = rand(size(test_x)) * rescale_fac;
    inp_image = spike_snapshot <= test_x;
    
    nn.layers{1}.spikes = inp_image;
    nn.layers{1}.sum_spikes = nn.layers{1}.sum_spikes + inp_image;
    for l = 2 : numel(nn.size)
        % Get input impulse from incoming spikes
        I = nn.layers{l-1}.spikes*nn.W{l-1}';

        C = 1./nn.layers{l}.E;
        dv = I./ C;
        
        % Add input to membrane p otential
        nn.layers{l}.mem = nn.layers{l}.mem + dv;
        % Check for spiking
        nn.layers{l}.spikes = nn.layers{l}.mem >= lifsim_opts.threshold;
        % Reset
        nn.layers{l}.mem(nn.layers{l}.spikes) = lifsim_opts.rest;
        % Ban updates until....
        nn.layers{l}.refrac_end(nn.layers{l}.spikes) = t + lifsim_opts.t_ref;
        % Store result for analysis later
        nn.layers{l}.sum_spikes = nn.layers{l}.sum_spikes + nn.layers{l}.spikes;
        
        y = nn.layers{l}.spikes ;
        delta_E = (1./nn.layers{l}.E + (y.*I + beta.*(1-y).*I)) / (lifsim_opts.threshold - lifsim_opts.rest).*learning_rate;
        nn.layers{l}.E = nn.layers{l}.E + delta_E;
    end
    
    if(mod(round(t/dt),round(lifsim_opts.report_every/dt)) == round(lifsim_opts.report_every/dt)-1)
        [~, guess_idx] = max(nn.layers{end}.sum_spikes');
        acc = sum(guess_idx==ans_idx)/size(test_y,1)*100;
        fprintf('Time: %1.3fs | Accuracy: %2.2f%%.\n', t, acc);
        nn.performance(end+1) = acc;
    else
        fprintf('.');
    end
end

[~, guess_idx] = max(nn.layers{end}.sum_spikes');
acc = sum(guess_idx==ans_idx)/size(test_y,1)*100;
fprintf('\nFinal spiking accuracy: %2.2f%%\n', acc);


% spike1 = reshape(nn.layers{1,1}.sum_spikes(2,:), 28, 28);
% figure;imagesc(spike1', [0, 10]);
% fprintf('\nsum_spikes of layer1 of evol-sfnn: %d\n', sum(sum(spike1)));
% spike2 = reshape(nn.layers{1,2}.sum_spikes(2,:), 40, 30);
% figure;imagesc(spike2, [0, 10]);
% fprintf('\nsum_spikes of layer2 of evol-sfnn: %d\n', sum(sum(spike2)));
% spike3 = reshape(nn.layers{1,3}.sum_spikes(2,:), 40, 30);
% figure;imagesc(spike3, [0, 10]);
% fprintf('\nsum_spikes of layer3 of evol-sfnn: %d\n', sum(sum(spike3)));
% spike4 = nn.layers{1,4}.sum_spikes(2,:);
% figure;imagesc(spike4', [0, 10]);
% fprintf('\nsum_spikes of layer4 of evol-sfnn: %d\n', sum(spike4));

% C2 = squeeze(C_his{2}(1,:,:));
% figure;imagesc(C2, [0, 1])
% C3 = squeeze(C_his{3}(1,:,:));
% figure;imagesc(C3, [0, 1]);
% C4 = squeeze(C_his{4}(1,:,:));
% figure;imagesc(C4);
% 
% dv2 = squeeze(dv_his{2}(1,:,:));
% figure;imagesc(dv2, [-3, 3])
% dv3 = squeeze(dv_his{3}(1,:,:));
% figure;imagesc(dv3, [-3, 3]);
% dv4 = squeeze(dv_his{4}(1,:,:));
% figure;imagesc(dv4);
end
