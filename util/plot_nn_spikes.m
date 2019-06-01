function plot_nn_spikes(nn)

digit_index = 2;

spikes = nn.layers{1}.sum_spikes(digit_index, :);
spikes = reshape(spikes, 28, 28);
figure;
imagesc(spikes', [0, 50])

spikes = nn.layers{2}.sum_spikes(digit_index, :);
spikes = reshape(spikes, 40, 30);
figure;
imagesc(spikes, [0, 1000])

spikes = nn.layers{3}.sum_spikes(digit_index, :);
spikes = reshape(spikes, 40, 30);
figure;
imagesc(spikes, [0, 1000])

spikes = nn.layers{4}.sum_spikes(digit_index, :);
spikes = reshape(spikes, 10, 1);
figure;
imagesc(spikes, [0, 1000])
end