function yuv_save(filename, Y,U,V , mode)

switch mode
    case 1 % replace file
        fid = fopen(filename, 'w');
        
    case 2 % append to file
        fid = fopen(filename, 'a');
        
    otherwise
        fid = fopen(filename, 'w');
end


%% YUV read
buf = uint8( reshape(Y.', [], 1)); % reshape
count=fwrite(fid,buf,'uchar');

buf = uint8( reshape(U.', [], 1)); % reshape
count=fwrite(fid,buf,'uchar');

buf = uint8(reshape(V.', [], 1)); % reshape
count=fwrite(fid,buf,'uchar');

fclose(fid);