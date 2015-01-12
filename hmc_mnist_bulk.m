function [ X_out ] = hmc_mnist_bulk( X, limit, samples, resample, coord, filters )
%HMC_MNIST_BULK Summary of this function goes here
%   Detailed explanation goes here

height = 28;
width = 28;
if nargin < 6
    filters = zeros(height*2+1, width*2+1, 1, 1);
    filters(:,:,1,1) = coulomb_filter(height, width);
end    
X_out = zeros(min(size(X,2), limit)*resample, samples*coord*size(filters, 4));
tic;
for k = 1 : min(size(X,2), limit)
    for t = 1 : resample
        for j = 1 : size(filters, 4)
            filter = squeeze(filters(:,:,1,j));
            retry = 3;
            while 0 < retry
                [x,y,u,v,z] = hmc_mnist_sampler(X(:,k),'max',samples,true,filter);
                if samples * 0.9 < size(x,1)
                    break;
                else
                    retry = retry - 1;
                end
            end
            a = [x/width,y/height]';
            b = [(u+width)/2/width,(v+height)/2/height]';
            %c = X((x-1)*height+y,k);
            s1 = a(:)';
            s2 = b(:)';
            s3 = ((z-max(z))/min(z-max(z)))';
            X_out(resample*(k-1)+t,...
                (j-1)*samples*coord+1:...
                (j-1)*samples*coord+size(s1,2)) = s1;
            X_out(resample*(k-1)+t,...
                (j-1)*samples*coord+samples*2+1:...
                (j-1)*samples*coord+samples*2+size(s2,2)) = s2;
            X_out(resample*(k-1)+t,...
                (j-1)*samples*coord+samples*4+1:...
                (j-1)*samples*coord+samples*4+size(s3,2)) = s3;
        end
    end
    if mod(k, 1000) == 0
        fprintf('*');
    end
    if mod(k, 10000) == 0
        fprintf(' ');toc;
    end
end

end

