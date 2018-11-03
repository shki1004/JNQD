
%% Features for liner regression
% Implemented by Sehwan Ki

function feat = All_test_feature(ori_img)
% total 14 features
[height,width] = size(ori_img);
N = 8;
col = round(width/N);
row = round(height/N);

feat = zeros(row,col,14);

for r = 1:row
    for c = 1:col
        % JNQD QP
        test_patch = ori_img(N*(r-1)+1:N*r,N*(c-1)+1:N*c);
        %feat(r,c,1) = Average_Luminance(test_patch);
        feat(r,c,1) = Michelson_contrast(test_patch);
        feat(r,c,2) = Standard_deviation(test_patch);
        %feat(r,c,4) = Skewness(test_patch);
        %feat(r,c,5) = Kurtosis(test_patch);
        %feat(r,c,6) = Edge_density(test_patch);
        %feat(r,c,7) = Entropy(test_patch);
        %feat(r,c,8) = Local_Entropy(test_patch);
        [feat(r,c,3),feat(r,c,4)] = Mag_spectra(test_patch);
        feat(r,c,5) = Sharpness_metric_SCI(test_patch);
    end
end

end
function out = Average_Luminance(ori_img) % feature 1
% Average Luminance intensity
ori_img = double(ori_img);
out = mean(ori_img(:));
end

function out = Michelson_contrast(ori_img) % feature 2
% Michelson contrast
ori_img = double(ori_img);
max_val = max(ori_img(:));
min_val = min(ori_img(:));

out = (max_val - min_val)/(max_val+min_val);
end

function out = RMS_contrast(ori_img,dist_img) % feature 3
% RMS contrast
ori_img = double(ori_img);
dist_img = double(dist_img);

[M,N] = size(ori_img);

diff_img = ori_img - dist_img;

avg_diff_img = mean(diff_img(:));

numerator = sqrt(sum(sum((diff_img - avg_diff_img*ones(M,N)).^2))/(M*N-1));
denominator = mean(ori_img(:));

out = 20*log10(numerator/denominator);
end

function out = Standard_deviation(ori_img) % feature 4
% Standard_deviation
ori_img = double(ori_img);
out = std(ori_img(:));
end

function out = Skewness(ori_img) % feature 5
% Skewness
ori_img = double(ori_img);
out = skewness(ori_img(:),1);
end

function out = Kurtosis(ori_img) % feature 6
% Kurtosis
ori_img = double(ori_img);
out = kurtosis(ori_img(:),1);
end

function out = Edge_density(ori_img) % feature 7
% Edge_density
ori_img = double(ori_img);
edge_img = edge(ori_img,'canny');

out = mean(edge_img(:));
end

function out = Entropy(ori_img) % feature 8
% Entropy
ori_img = uint8(ori_img);

out = entropy(ori_img);
end

function out = Local_Entropy(ori_img) % feature 9
% Local Entropy
ori_img = uint8(ori_img);
ori_img_col = im2col(ori_img,[7,7]);
local_entropy = zeros(length(ori_img_col(1,:)),1);
for i = 1:length(ori_img_col(1,:))
    local_entropy(i) = entropy(ori_img_col(:,i));
end

out = mean(local_entropy);
end

function [slope,intercept] = Mag_spectra(ori_img)
% Magnitude of spectra
% slope
% intercept
persistent N;
persistent wnd;

if (nargin == 0)
    N = [];
    wnd = [];
    return;
end

if (nargin == 2 || isempty(N))
    N = size(ori_img, 1);
    wnd = hanning(N);
    wnd = wnd * wnd';
end

if (~isa(ori_img, 'double'))
    ori_img = double(ori_img);
end
ori_img_wnd_prod = ori_img .* wnd;

[fs, as] = eo_polaraverage(abs(fft2(ori_img_wnd_prod)));
fs = fs(1:end);
as = as(1:end);

p = polyfit(log(fs), log(as), 1);
slope = -p(1);
intercept = p(2);

end % feature 10, 11

function out = Sharpness_metric_S3(ori_img) % feature 12
% Sharpness Metric : S3
ori_img = double(ori_img);
[s_map1 s_map2 s3] = s3_map(ori_img)
out = s3;
end

function out = Sharpness_metric_LBP(ori_img) % feature 13
% Sharpness Metric : LBP
ori_img = double(ori_img);
s = 8;
threshold = 0.016;  %T_lbp
out = lbpSharpness(ori_img,s,threshold);
end

function out = Sharpness_metric_SCI(ori_img) % feature 14
% Sharpness Metric : SCI
ori_img = double(ori_img);
N = 8;
load('spatial_freq.mat');
[tau_sci,kurtosis,Ct] = SCI(ori_img,N,w);
out = mean(tau_sci(:));
end

% Functions for Sharpness metric(S3) and Magnitude of spectra
function [x y] = eo_polaraverage(data)

% conversion cartesian to polar and average from 0 [rad] to rr [rad] step
% rr/dr [rad]
% return value excludes average and includes nyquist frequency spectrum
%
% usage:
% s = eo_polaraverage(fftdata)
% [f s] = eo_polaraverage(fftdata)

% $Revision: 1.1 $
% $Date: 2006/08/07 02:57:43 $
% $Author: kannon $

rr = 2*pi; dr = 360;

n = length(data);
% average = data(1,1);
data(1,1) = (data(2,1)+data(1,2))/2;

%for r = 0:(n/2-1)
for r = 1:(n/2) % for each polar frequency
    zs = 0;
    for ith = 0:(dr-1) % for each 1 degree angle
        th = ith/dr; % convert to radian
        x = r * sin (th*rr); % x coordinate
        y = r * cos (th*rr); % y coordinate
        
        x1 = sign(x) * floor ( abs (x) ); % rounding
        x2 = sign(x) * ceil  ( abs (x) );
        y1 = sign(y) * floor ( abs (y) );
        y2 = sign(y) * ceil  ( abs (y) );
        
        ex = abs(x - x1);
        ey = abs(y - y1);
        
        if(x2<0)
            ex = abs(x - x2);
            if(x1<0)
                x1 = n + x1;
            end
            x2 = n + x2;
        end
        
        if(y2<0)
            ey = abs(y - y2);
            if(y1<0)
                y1 = n + y1;
            end
            y2 = n + y2;
        end
        
        f11 = data(x1+1, y1+1);
        f12 = data(x1+1, y2+1);
        f21 = data(x2+1, y1+1);
        f22 = data(x2+1, y2+1);
        
        %z = interp2([0 1;0 1], [0 0;1 1], [f11 f21;f12 f22], ex, ey, 'linear');
        z = (f21-f11)*ex*(1-ey) + (f12-f11)*(1-ex)*ey + (f22-f11)*ex*ey + f11;
        
        zs = zs + z;
    end
    s(r+1) = zs/dr;
end

f = linspace(0,0.5,length(s));
s = s(2:end);
f = f(2:end);

if(nargout>=2)
    x = f;
    y = s;
else
    x = s;
end
end

function [s_map1 s_map2 s3] = s3_map(img, show_res)
% Input: img is a gray scale image, in double type, range from 0 - 255.
% You have to convert to gray scale if your image
% is color. You also have to cast img to double in order to run this code
% Parameter show_res = 1 to show results
% Output:
% s_map1: The sharpness map measure based on spectral slope
% s_map2: The sharpness map measure based on total variation (spatial)
% s3: the final sharpness map (combination of s_map1 and s_map2)
if nargin < 2
    show_res = 0;
end

% ----------------------------------------------------------------------
% blr_map1
s_map1 = spectral_map(img, 16);

%-----------------------------------------------------------------------
% blr_map2
s_map21 = spatial_map(img, 8); % Spatial map, blocks start from (1,1)
s_map22 = spatial_map(img, 4); % Spatial map, blocks start from (5,5)
s_map2 = max(s_map21, s_map22);

%-----------------------------------------------------------------------
% combine
s_map1(s_map1 < -99) = 0;
s_map2(s_map2 < -99) = 0;

alpha = 0.5;
s3 = (s_map1.^alpha) .* ((s_map2).^(1-alpha));
if show_res
    %     figure; imshow(s_map1);
    %     figure; imshow(s_map2);
    %     figure; imshow(img/255);
    %     figure; imshow(s3);
end
end %function

% Spectral Sharpness, slope of power spectrum
function res = spectral_map(img, pad_len)
blk_size = 8; %big block size for more coefficients of the power spectrum
d_blk = blk_size/4; % Distance b/w blocks

pad_L = fliplr(img(:, 1:pad_len)); % Take 16 columns on the left of the
% original image to pad to the left
pad_R = fliplr(img(:, end-pad_len:end));%Take 16 columns on the right of the
% original image to pad to the
% right
img = [pad_L img pad_R]; %Pad left and right

pad_T = flipud(img(1:pad_len, :)); %Similarly, pad top and bottom
pad_B = flipud(img(end-pad_len:end, :));
img = [pad_T; img; pad_B];

num_rows = size(img, 1);
num_cols = size(img, 2);
res = zeros(num_rows, num_cols) - 100;
contrast_thresold = 0;

disp_progress; % Just to show progress
for r = blk_size/2+1:d_blk:num_rows-blk_size/2 % Just start from inside blocks
    % of the padded image
    disp_progress(r, num_rows);
    for c = blk_size/2+1:d_blk:num_cols-blk_size/2
        gry_blk = img(...
            r-blk_size/2:r+blk_size/2-1,...
            c-blk_size/2:c+blk_size/2-1 ...
            );
        contrastMap = contrast_map_overlap(gry_blk);
        if(max(contrastMap(:))> contrast_thresold) % Avoid the case when contrast = 0
            val = blk_amp_spec_slope_eo_toy(gry_blk); % Val(1) will be the slope of
            % power spectrum of the block
            val(1) = 1 - 1 ./ (1 + exp(-3*(val(1) - 2))); %Input to a sigmoid function
            %if(max(gry_blk(:))==min(gry_blk(:))) % Black block
            % val_1 = 0;
            %else
            val_1 = val(1);
            %end
        else
            val_1 = 0;
        end
        res(...
            r-d_blk/2:r+d_blk/2-1,...
            c-d_blk/2:c+d_blk/2-1 ...
            ) = val_1;
    end
end
% Remove padded parts
res = res(pad_len+1:end-pad_len-1, pad_len+1:end-pad_len-1);
end % function

% Spatial Sharpness, local total variation
function res = spatial_map(img, pad_len)
% pad_len = 8 if we dont want to shift img
% pad_len = 4 if we want to shift img by 4;
blk_size = 32;

pad_L = fliplr(img(:, 1:pad_len)); % Take pad_len columns on the left of
% the original image to pad to the left
pad_R = fliplr(img(:, end-pad_len:end));%Take pad_len columns on the right
% of the original image to pad to the right
img = [pad_L img pad_R]; %Pad left and right

pad_T = flipud(img(1:pad_len, :)); %Similarly, pad top and bottom
pad_B = flipud(img(end-pad_len:end, :));
img = [pad_T; img; pad_B];

[num_rows, num_cols] = size(img);
res = zeros(num_rows, num_cols);

for r = blk_size/2+1 : blk_size : num_rows-blk_size/2
    for c = blk_size/2+1 : blk_size : num_cols-blk_size/2
        gry_blk = img(...
            r-blk_size/2 : r+blk_size/2-1,...
            c-blk_size/2 : c+blk_size/2-1 ...
            );
        % Measure local total variation for every 2x2 block of gry_blk
        tmp_idx = 1;
        for i = 1 : blk_size - 1
            for j = 1 : blk_size - 1
                tv_tmp(tmp_idx) = (abs(gry_blk(i,j) - gry_blk(i,j+1))...
                    + abs(gry_blk(i,j) - gry_blk(i+1,j))...
                    + abs(gry_blk(i,j) - gry_blk(i+1,j+1))...
                    + abs(gry_blk(i+1,j+1) - gry_blk(i+1,j))...
                    + abs(gry_blk(i+1,j) - gry_blk(i,j+1))...
                    + abs(gry_blk(i+1,j+1) - gry_blk(i,j+1)))/255; %Each pixel ranges
                %from 0 - 255, so divide by 255 to make it from 0 - 1
                tmp_idx = tmp_idx + 1;
            end
        end
        
        tv_max = max(tv_tmp) / 4; % Normalize tv_max to be from 0 -1. We can
        % easily see that the maximum value of total
        % variation for each 2x2 block is 4, in
        % blocks like   1 0
        %               0 1
        
        res(...
            r - blk_size/2 : r + blk_size/2-1,...
            c - blk_size/2 : c + blk_size/2-1 ...
            ) = tv_max;
    end
end
res = res(pad_len + 1 : end-pad_len - 1, pad_len + 1 : end-pad_len - 1);

end % function

function disp_progress(p, p_max)

persistent p_last;
if (nargin == 0)
    p_last = [];
    fprintf(1, '%s\n', '');
    return;
end

p_done = p / p_max * 100;
p_done = round(p_done / 10) * 10;

%[p_done p_last]

if (p_done == p_last)
    return;
end

if (~isempty(p_last))
    %  fprintf(1, '%d\n', p_last);
    fprintf(1, '%s\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b', '');
    %  return;
end
p_last = p_done;

switch (p_done)
    case 0
        fprintf(1, '%s', '[          ] 0%  ');
    case 10
        fprintf(1, '%s', '[|         ] 10% ');
    case 20
        fprintf(1, '%s', '[||        ] 20% ');
    case 30
        fprintf(1, '%s', '[|||       ] 30% ');
    case 40
        fprintf(1, '%s', '[||||      ] 40% ');
    case 50
        fprintf(1, '%s', '[|||||     ] 50% ');
    case 60
        fprintf(1, '%s', '[||||||    ] 60% ');
    case 70
        fprintf(1, '%s', '[|||||||   ] 70% ');
    case 80
        fprintf(1, '%s', '[||||||||  ] 80% ');
    case 90
        fprintf(1, '%s', '[||||||||| ] 90% ');
    case 100
        fprintf(1, '%s', '[||||||||||] 100% ');
end
drawnow;

end

function [cnt_map] = contrast_map_overlap(img)
%input must be double, range 0:255
img_lum = (0.7656 + 0.0364*img).^2.2;

blk_size = 8;
d_blk = blk_size/2;

[num_rows, num_cols] = size(img_lum);

cnt_map = zeros(num_rows, num_cols);

for r = 1:d_blk:num_rows-d_blk
    for c = 1:d_blk:num_cols-d_blk
        
        rs = r:r+blk_size-1;
        cs = c:c+blk_size-1;
        rs1 = r:r+d_blk-1;
        cs1 = c:c+d_blk-1;
        blk = img_lum(rs, cs);
        m_lum = mean2(blk);
        if m_lum > 127.5
            blk = 255 - blk;
            m_lum = mean2(blk);
        end
        if (m_lum > 2 && max(blk(:))-min(blk(:)) > 5)
            contrast = std2(blk) / m_lum; % Using rms contrast only when a block
            % has enough brightness and some
            % variant ...
        else
            contrast = 0; % otherwise set to 0
        end
        if(contrast > 5)
            contrast = 5;
        end
        cnt_map(rs1, cs1) = contrast/5;
    end
end
end

% Fnctions for Sharpness metric(LBP)
function sharpnessMap = lbpSharpness(im, s, threshold)

if (size(im,3)==3)
    im_gray = rgb2gray(im);
else
    im_gray = im;
end

[height, width, ~] = size(im);

window_r = (s-1)/2;

num = s^2;

lbpmap = lbpCode(im_gray,threshold*255);

lbp_map_pad = padarray(lbpmap,[window_r,window_r],'replicate');

% naive implementation
map = zeros(height,width);
for j = 1:height
    for i = 1:width
        lbpmap_patch = lbp_map_pad(j:j+s-1,i:i+s-1);
        temp = (lbpmap_patch>=6);
        map(j,i) = sum(temp(:))/num;
        
        
    end
end


sharpnessMap = norm_patch(map);
end

function out = lbpCode(I, threshold)
interpOff = sqrt(2)/2;
I = double(I);

P = padarray(I, [1 1], 'replicate');

% Entry (i,j) of this matrix contains the pixel to the
% right of (i,j) in I.
right = P(2:end-1, 3:end);

% Meanings of these are similar to that of 'right', above.
left = P(2:end-1, 1:end-2);
above = P(1:end-2, 2:end-1);
below = P(3:end, 2:end-1);
aboveRight = P(1:end-2, 3:end);
aboveLeft = P(1:end-2, 1:end-2);
belowRight = P(3:end, 3:end);
belowLeft = P(3:end, 1:end-2);

% Entry (i,j) of interpK contains the interpolated value of g_k for
% pixel (i,j) of the original image.
interp0 = right;
interp1 = (1-interpOff)*((1-interpOff) .* I + interpOff .* right) + interpOff *((1-interpOff) .* above + interpOff .* aboveRight);
interp2 = above;
interp3 = (1-interpOff)*((1-interpOff) .* I + interpOff .* left ) + interpOff *((1-interpOff) .* above + interpOff .* aboveLeft);
interp4 = left;
interp5 = (1-interpOff)*((1-interpOff) .* I + interpOff .* left ) + interpOff *((1-interpOff) .* below + interpOff .* belowLeft);
interp6 = below;
interp7 = (1-interpOff)*((1-interpOff) .* I + interpOff .* right ) + interpOff *((1-interpOff) .* below + interpOff .* belowRight);



interp0 = floor(interp0);
interp1 = floor(interp1);
interp2 = floor(interp2);
interp3 = floor(interp3);
interp4 = floor(interp4);
interp5 = floor(interp5);
interp6 = floor(interp6);
interp7 = floor(interp7);
% Image s_k at (i,j) contains the bit for g_k at (i,j)
s0 = s(interp0 - I-threshold);
s1 = s(interp1 - I-threshold);
s2 = s(interp2 - I-threshold);
s3 = s(interp3 - I-threshold);
s4 = s(interp4 - I-threshold);
s5 = s(interp5 - I-threshold);
s6 = s(interp6 - I-threshold);
s7 = s(interp7 - I-threshold);




% Compute the uniformity.
U = abs(s0 - s7) + ...
    abs(s1 - s0) + ...
    abs(s2 - s1) + ...
    abs(s3 - s2) + ...
    abs(s4 - s3) + ...
    abs(s5 - s4) + ...
    abs(s6 - s5) + ...
    abs(s7 - s6);

% Compute number of bits in each LBP.  For the uniform
% patterns this is the correct pattern id.
LBP81riu2 = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7;

% If the pattern is not uniform, replace the bit count
% with 9, to indicate a non-uniform pattern.
LBP81riu2(U > 2) = 9;


out = LBP81riu2;
end

function sVal = s(x)
sVal = x >0;
end

% Functions for Sharpeness metric(SCI)
function [tau_sci,kurtosis,Ct] = SCI(img_L,N,w) %TIP 2014 CM-JND model

[height, width] = size(img_L);

C = blkproc(img_L,[N N],'dct2');

col = round(width/N);
row = round(height/N);

m2 = zeros(row,col);
m4 = zeros(row,col);
normalize = zeros(row,col);

K = 255;

for i = 1:col
    for j = 1:row
        
        for x = 1:N
            for y = 1:N
                if(x == 1 && y == 1)
                else
                    m2(j,i) = m2(j,i) + (w(y,x)^2)*((C((j-1)*N+y,(i-1)*N+x)).^2);
                    m4(j,i) = m4(j,i) + (w(y,x)^4)*((C((j-1)*N+y,(i-1)*N+x)).^2);
                    normalize(j,i) = normalize(j,i) + (C((j-1)*N+y,(i-1)*N+x)).^2;
                end
            end
        end
        
    end
end
tau_sci = zeros(row,col);
kurtosis = zeros(row,col);
Ct = zeros(row,col);
for i = 1:row
    for j = 1:col
        if(normalize(i,j) == 0 || m2(i,j) == 0 || m4(i,j) == 0)
            tau_sci(i,j) = 0;
        else
            kurtosis(i,j) = normalize(i,j)*m4(i,j)/(m2(i,j)^2);
            Ct(i,j) = sqrt(normalize(i,j))/(N*(K/2));
            tau_sci(i,j) = kurtosis(i,j)^(-0.7)*Ct(i,j)^(1.4);
        end
    end
end

end





