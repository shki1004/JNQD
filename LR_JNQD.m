%% Learning-based just-noticeable-quantization-distortion modeling for perceptual video coding
% TIP 2018
% Sehwan Ki
% shki@kaist.ac.kr


%% Linear Regression based JNQD test code
clear;clc;close all;

load('spatial_freq.mat'); % spatial frequency with viewing distance = 1.3m

load('gradient_offset_weight_qp_intra.mat'); % Trained gradient and slope values

sequence = {'BQMall'};

QP = [22,27,32,37];
frame = [600];
H = [480];
W = [832];

for seq = 1:1
    fprintf('seq : %s \n', strjoin(sequence(seq)));
    
    height = H(seq);
    width = W(seq);
    
    N = 8; % dct block size
    
    col = round(width/N);
    row = round(height/N);
    
    for fn = 1:frame(seq)
        
        fprintf('Frame : %d \n', fn);
        % 1. Input sequence
        str = sprintf('test_sequences/%s_%dx%d_%d.yuv', strjoin(sequence(seq)), W(seq), H(seq), floor(frame(seq)/10));
        [Y,U,V] = yuv_load(str, W(seq), H(seq), fn);
        Y_ori = double(Y);
        
        % 2. Calculate ERJND
        C_ori = blkproc(Y_ori,[8 8],'dct2'); % original block DCT Coefficient
        
        for r = 1:row
            for c = 1:col
                Y_ori_patch = Y_ori(N*(r-1)+1:N*r,N*(c-1)+1:N*c);
                C_ori_patch = C_ori(N*(r-1)+1:N*r,N*(c-1)+1:N*c);
                % ERJND_ori
                ERJND_ori(N*(r-1)+1:N*r,N*(c-1)+1:N*c) = block_ERJND(Y_ori_patch,w,C_ori_patch);
            end
        end
        
        %% 3-1. Gradient & Offset ¸¦ regression
        % feature extraction
        feat = All_test_feature(Y_ori);
        
        feature = [];
        for i = 1:5 % Using top-correlated 5 features
            Temp = feat(:,:,i);
            feature(i,:) = Temp(:);
        end
        feature(2,:) = log(feature(2,:)+eps);
        feature(5,:) = log(feature(5,:)+eps);
        feature(6,:) = ones(1,length(feature(1,:)));
        
        for qp = 1:4
            
            % 3-2. Linear regression for gradient & offset 
            % gradient & offset regression
            if(qp == 1)
                regress_gradient = (feature')*weight22_gradient;
                regress_offset = (feature')*weight22_offset;
            elseif(qp == 2)
                regress_gradient = (feature')*weight27_gradient;
                regress_offset = (feature')*weight27_offset;
            elseif(qp == 3)
                regress_gradient = (feature')*weight32_gradient;
                regress_offset = (feature')*weight32_offset;
            elseif(qp == 4)
                regress_gradient = (feature')*weight37_gradient;
                regress_offset = (feature')*weight37_offset;
            end
            
            %4. Determine an optimal alpha
            alpha = [];
            for i = 1:length(regress_gradient(:))
                if(regress_gradient(i) == 0)
                    if(regress_offset(i) > 1)
                        alpha(i) = 0;
                    else
                        alpha(i) = 1;
                    end
                else
                    % Assume that CDVM is linear function
                    val = (1 - regress_offset(i))/regress_gradient(i);
                    if(val < 0)
                        alpha(i) = 0;
                    elseif(val > 1)
                        alpha(i) = 1;
                    else
                        alpha(i) = val;
                    end
                end
            end
            alpha = reshape(alpha,[row,col]);
            
            % 5. Calculate LR-JNQD
            for r = 1:row
                for c = 1:col
                    alpha_image(N*(r-1)+1:N*r,N*(c-1)+1:N*c) = alpha(r,c);
                end
            end
            
            alpha_image(find(isnan(alpha_image)==1)) = 0; % Remove nan values
            
            LR_JNQD_val = alpha_image.*ERJND_ori;
            
            % 6. Preprocessing using LR-JNQD
            C_pre = sign(C_ori).*max(abs(C_ori)-LR_JNQD_val,0);
            Y_pre = blkproc(C_pre,[8 8],'idct2');
            Y_pre = uint8(Y_pre);
            
            if(qp == 1)
                alpha_qp22_I = alpha;
            elseif(qp == 2)
                alpha_qp27_I = alpha;
            elseif(qp == 3)
                alpha_qp32_I = alpha;
            elseif(qp == 4)
                alpha_qp37_I = alpha;
            end
            if(qp == 1)
                % QP 22
                filename = sprintf('%s/%s_QP22_LR_JNQD_all.yuv',strjoin(sequence(seq)),strjoin(sequence(seq)));
            elseif(qp == 2)
                % QP 27
                filename = sprintf('%s/%s_QP27_LR_JNQD_all.yuv',strjoin(sequence(seq)),strjoin(sequence(seq)));
            elseif(qp == 3)
                % QP 32
                filename = sprintf('%s/%s_QP32_LR_JNQD_all.yuv',strjoin(sequence(seq)),strjoin(sequence(seq)));
            elseif(qp == 4)
                % QP 37
                filename = sprintf('%s/%s_QP37_QP_adapted_JNQD_all.yuv',strjoin(sequence(seq)),strjoin(sequence(seq)));
            end
            
            if(fn == 1)
                mode = 1;
                yuv_save(filename, Y_pre,U,V, mode);
            else
                mode = 2;
                yuv_save(filename, Y_pre,U,V, mode);
            end
            
        end
    end
end
