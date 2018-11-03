%% Learning-based just-noticeable-quantization-distortion modeling for perceptual video coding
% TIP 2018
% Sehwan Ki
% shki@kaist.ac.kr

%% CNN-JNQD test code

% matconvnet setup
%setup() ;
setup('useGpu', true);

clear;

% test sequences
seq = {'BQMall_832x480_60'};
frame = [600];
Height = [480];
Width = [832];

for ind = 1:1
    for qp = [22,27,32,37]
        % CNN_JNQD_(qp)_(filter size)_(channel size)_(input size)_(# of layers)_(batchsize)_(learning rate)                
        str = sprintf('CNN-JNQD_trained_wieghts/CNN_JNQD_%d_3_64_14_3_128_10^(-2)/net-epoch-300.mat',qp);
        load(str);

        % Deploy: remove loss
        net.layers(end) = [] ;
        tic
        for fn = 1:frame(ind)
            fprintf('QP = %d, EPOCH = %d, Frame = %d \n',qp, EPOCH,fn);

            height = Height(ind);
            width = Width(ind);
            str = sprintf('test_sequence/%s.yuv', char(seq(ind)));

            [Y,U_ori,V_ori] = yuv_load(str, width, height, fn);
            im_gray = double(Y);
            ori_im = single(im_gray/255);

            % zero padding
            ori_im_padding = padarray(ori_im,[3,3],'replicate');

            % Testing
            res = vl_simplenn(net, ori_im_padding) ;
            CNN_out = res(end).x;

            CNN_jndqd_im = CNN_out;
            CNN_jndqd_im = CNN_jndqd_im*255;

            CNN_jndqd_im_clipping = CNN_jndqd_im;

            for i = 1:length(CNN_jndqd_im(:,1))
                for j = 1:length(CNN_jndqd_im(1,:))
                    if(CNN_jndqd_im(i,j) > 255)
                        CNN_jndqd_im_clipping(i,j) = 255;
                    elseif(CNN_jndqd_im(i,j) < 0)
                        CNN_jndqd_im_clipping(i,j) = 0;
                    end
                end
            end

            filename = sprintf('CNN_JNQD_result/%s_%d.yuv',char(seq(ind)),qp);


            if(fn == 1)
                mode = 1;
                yuv_save(filename, uint8(CNN_jndqd_im_clipping),U_ori,V_ori, mode);
            else
                mode = 2;
                yuv_save(filename, uint8(CNN_jndqd_im_clipping),U_ori,V_ori, mode);
            end
        end
        toc
        
    end
end



