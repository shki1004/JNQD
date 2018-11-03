
%% Cal ERJND
function [ERJND] = block_ERJND(blockimg,w,C)

N = 8; 
% feature 1 : SCI
[sci_val] = SCI_fast(blockimg,N,w,C); % Calculate Block SCI value

ERJND = 42.5*sci_val.^(0.54);

end