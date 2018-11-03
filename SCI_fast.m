
%% Calculate SCI value
% TIP 2014 CM-JND model
% Implemented by Sehwan Ki

function [tau_sci,kurtosis,Ct] = SCI_fast(img_L,N,w,C) 

[height, width] = size(img_L);

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

