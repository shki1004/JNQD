
%% Learning-based just-noticeable-quantization-distortion modeling for perceptual video coding
% TIP 2018
% Sehwan Ki
% shki@kaist.ac.kr

clear;close all;clc
%% CNN-JNQD training code

% matconvnet setup
%setup() ;
setup('useGpu', true);

% Parameter setting
bat = 128;
lr = 10^-2;
epoch_num = 300;

filter_size = 3;
filter_channel = 64;
layer = 3;

for QP = [22,27,32,37]
    for size_input = 14
        % set input param
        stride = 8;
        size_input_in = size_input;
        size_label_in = 8;

        fn_in(1,:) = [filter_size, 1, filter_channel]; 
        for layer_n = 2:(layer-1)
            fn_in(layer_n,:) = [filter_size, filter_channel, filter_channel];
        end
        fn_in(layer,:) = [filter_size, filter_channel, 1];


        if(QP == 22)
            [data, label] = generate_img_data('Train/ori','Train/jnqd/QP22',size_input_in, size_label_in,stride); % data
        elseif(QP == 27)
            [data, label] = generate_img_data('Train/ori','Train/jnqd/QP27',size_input_in, size_label_in,stride); % data
        elseif(QP == 32)
            [data, label] = generate_img_data('Train/ori','Train/jnqd/QP32',size_input_in, size_label_in,stride); % data
        elseif(QP == 37)
            [data, label] = generate_img_data('Train/ori','Train/jnqd/QP37',size_input_in, size_label_in,stride); % data
        end

        [imdb] = generate_imdb(data,label); %generate imdb.mat

        net = initializeCNN_SR_multi(size_input,layer,fn_in);

        % Display network
        vl_simplenn_display(net) ;

        % Add a loss (using a custom layer)
        net = addCustomLossLayer(net, @l2LossForward, @l2LossBackward) ;

        %% Training
        % Train
        trainOpts.expDir = sprintf('CNN-JNQD_trained_wieghts/CNN_JNQD_%d_%d_%d_%d_%d_%d_10^(-%d)',QP,filter_size,filter_channel,size_input,layer,bat,log10(lr)) ;
        trainOpts.gpus = 1 ;
        trainOpts.batchSize = bat ;
        trainOpts.learningRate = lr;
        trainOpts.plotDiagnostics = false ;
        %trainOpts.plotDiagnostics = true ; % Uncomment to plot diagnostics
        trainOpts.numEpochs = epoch_num ;
        trainOpts.errorFunction = 'none' ;

        net = cnn_train(net, imdb, @getBatch, trainOpts) ;

    end
end

