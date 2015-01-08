function [ X_out ] = hmc_mnist_bulk( X, limit, samples, resample, coord )
%HMC_MNIST_BULK Summary of this function goes here
%   Detailed explanation goes here

height = 28;
width = 28;
X_out = zeros(min(size(X,2), limit)*resample, samples*coord);
tic;
for k = 1 : min(size(X,2), limit)
    for t = 1 : resample
        [x,y,u,v,z] = hmc_mnist_sampler(X(:,k),'max',samples+1,true);
        a = [x/width,y/height]';
        b = [(u+width)/2/width,(v+height)/2/height]';
        %c = X((x-1)*height+y,k);
        s = [a(:)', b(:)', ((z-max(z))/min(z-max(z)))'];
        X_out(resample*(k-1)+t,1:size(s,2)) = s;
    end
    if mod(k, 1000) == 0
        fprintf('*');
    end
    if mod(k, 10000) == 0
        fprintf(' ');toc;
    end
end

end

