function [ X_out ] = hmc_mnist_bulk( X, limit, resample )
%HMC_MNIST_BULK Summary of this function goes here
%   Detailed explanation goes here

coord = 4;
samples = 49;
X_out = zeros(min(size(X,2), limit)*resample, samples*coord);
tic;
for k = 1 : min(size(X,2), limit)
    for t = 1 : resample
        [x,y,u,v] = hmc_mnist_sampler(X(:,k),'max',samples+1,true);
        a = [x,y]';
        b = [u,v]';
        s = [a(:)'/28, (b(:)'+28)/56];
        X_out(resample*(k-1)+t,:) = zeros(1, 28^2/coord);
        X_out(resample*(k-1)+t,1:size(s,2)) = s;
    end
    if mod(k, 1000) == 0
        fprintf('*');
    end
    if mod(k, 10000) == 0
        toc;
    end
end

end

