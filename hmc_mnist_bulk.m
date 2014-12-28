function [ X_out ] = hmc_mnist_bulk( X, limit )
%HMC_MNIST_BULK Summary of this function goes here
%   Detailed explanation goes here

% 4 times sampling with 1/4 size
resample = 4;
coord = 4;
X_out = zeros(min(size(X,2), limit)*resample, 28^2/resample);
tic;
for k = 1 : min(size(X,2), limit)
    for t = 1 : 4
        [x,y,u,v] = hmc_mnist_sampler(X(:,k),'max',28^2/resample/coord+1,true);
        a = [x,y]';
        b = [u,v]';
        s = [a(:)'/28, (b(:)'+28)/56];
        X_out(4*(k-1)+t,:) = zeros(1, 28^2/resample);
        X_out(4*(k-1)+t,1:size(s,2)) = s;
    end
    if mod(k, 1000) == 0
        fprintf('*');
    end
    if mod(k, 10000) == 0
        toc;
    end
end

end

