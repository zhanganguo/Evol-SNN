function plot_cnn_spikes(cnn)

digit_index = 2;

input = cnn.layers{1}.sum_spikes{1}(:,:,digit_index);
fprintf('\nsum_spikes of layer1 of scnn: %d\n', sum(sum(input)));
sum_spikes = 0;
for i = 1 : numel(cnn.layers{2}.sum_spikes)
    conv1{i} = cnn.layers{2}.sum_spikes{i}(:,:,digit_index);
    sum_spikes = sum_spikes + sum(sum(conv1{i}));
end
fprintf('\nsum_spikes of layer2 of scnn: %d\n', sum_spikes);
sum_spikes = 0;
for i = 1 : numel(cnn.layers{4}.sum_spikes)
    conv2{i} = cnn.layers{4}.sum_spikes{i}(:,:,digit_index);
    sum_spikes = sum_spikes + sum(sum(conv2{i}));
end
fprintf('\nsum_spikes of layer3 of scnn: %d\n', sum_spikes);
output = cnn.o_sum_spikes(:,digit_index);
fprintf('\nsum_spikes of layer4 of scnn: %d\n', sum(output));

cc1 = ones(24*3+5, 24*4+4)*10;
for i = 1 : 3
    for j = 1 : 4
        cc1(1+i+(i-1)*24:i*24+i, 1+j+(j-1)*24:j*24+j) = conv1{(i-1)*3+j};
    end
end

cc2 = ones(8*8+9, 8*8+9)*10;
for i = 1 : 8
    for j = 1 : 8
        cc2(1+i+(i-1)*8:i*8+i, 1+j+(j-1)*8:j*8+j) = conv2{(i-1)*8+j};
    end
end

% figure;
% imagesc(input', [0, 10]);
% figure;
% imagesc(cc1', [0, 10]);
% figure;
% imagesc(cc2', [0, 10]);
% figure;
% imagesc(output, [0, 10]);
