function [Y,U,V]=yuv_load(filename, width, height, idxframe)

% input
% - filename : file name of yuv file
% - height : Heigth of image = 288
% - width : Width of image = 352
% - idxframe : wanted frame index

% output
% - yuv  : YUV image

fid=fopen(filename,'r'); % file open

size=1.5*width*height; % open image size 
fseek(fid,(idxframe-1)*size,'bof');

%% YUV read 
Y=fread(fid,width*height,'uchar');
Y=uint8(reshape(Y,width,height)');

U=fread(fid,width*height/4,'uchar');
U=uint8((reshape(U,width/2,height/2))');

V=fread(fid,width*height/4,'uchar');
V=uint8((reshape(V,width/2,height/2))');

%  yuv(:,:,1)=Y;
%  yuv(:,:,2)=U;
%  yuv(:,:,3)=V;

fclose(fid);