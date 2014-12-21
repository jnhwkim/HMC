function [ X_out ] = hmc_mnist_bulk( X, limit )
%HMC_MNIST_BULK Summary of this function goes here
%   Detailed explanation goes here

X_out = zeros(min(size(X,2), limit), 4*196+28*28);
tic;
for k = 1 : min(size(X,2), limit)
    [x,y,u,v] = hmc_mnist_sampler(X(:,k));
    a = [x,y]';
    b = [u,v]';
    X_out(k,:) = [a(:)'/28, (b(:)'+28)/56, X(:,k)'];
    
    if mod(k, 1000) == 0
        fprintf('*');
    end
    if mod(k, 10000) == 0
        toc;
        fprintf('\n');
    end
end

end

