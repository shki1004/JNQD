function [data, label] = generate_img_data(folder1,folder2,size_input, size_label,stride)
%stride = 8;

%% initialization
data = zeros(size_input, size_input, 1, 1);
label = zeros(size_label, size_label, 1, 1);
padding = abs(size_input - size_label)/2;
count_label=0;
count_data=0;

%% generate data
filepaths = dir(fullfile(folder1,'*.yuv')); % original yuv --> Input

filepaths2 = dir(fullfile(folder2,'*.yuv')); % JNQD yuv --> target

fn = 1;
width = 832;
height = 480;

for i = 1 : length(filepaths)
    for j= 1:4 %data augmentation
        for w = 1:2 %data augmentation
            [Y,U,V] = yuv_load(fullfile(folder1,filepaths(i).name), width, height, fn);
            ori_im = double(Y);
            ori = ori_im/255; % -> 0-1 normalization
            
            image = im2single(ori);
            
            if w == 1 %data augmentation
                image = flip(image,2); %data augmentation
            end %data augmentation
            image = imrotate(image,90*j);
            
            [hei,wid] = size(image);
            
            for x = 1 : stride : hei-size_input+1
                for y = 1 :stride : wid-size_input+1
                    count_data = count_data+1;
                    data(:, :, 1, count_data) = image(x : x+size_input-1, y : y+size_input-1);
                end
            end
            
            [Y,U,V] = yuv_load(fullfile(folder2,filepaths2(i).name), width, height, fn);
            jnqd_im = double(Y);
            jnqd = jnqd_im/255; % -> 0-1 normalization
            
            %% target
            im_label = im2single(jnqd);
            
            if w == 1 %data augmentation
                im_label = flip(im_label,2); %data augmentation
            end %data augmentation
            im_label = imrotate(im_label,90*j);
            
            for x = 1 : stride : hei-size_input+1
                for y = 1 :stride : wid-size_input+1
                    count_label = count_label+1;
                    label(:, :, 1, count_label) = im_label(x+padding : x+padding+size_label-1, y+padding : y+padding+size_label-1);
                end
            end
            
        end
    end
end

end