function [ x,y,u,v,X1,px,py ] = hmc_mnist_sampler( X )
%HMC_MNIST_SAMPLER Summary of this function goes here
%   Detailed explanation goes here

% Hamilton Monte carlo sampling parameters
epsilon = .25;
L = 15;
K = .5;
InfD = .1;
NUM_SAMPLES = 10;

I = reshape(X, [28 28]);

%% Initialize first gaze
q = [0 0]';
f_size = 7;
f_mask = fspecial('gaussian', [f_size f_size], 0.5) - ...
         ones(f_size,f_size) / f_size^2;
X0 = conv2(I, f_mask);
%f0 = figure(4);
%set(f0, 'Position', [300 0 300 300]);
%imshow(X0*30, jet(36), 'InitialMagnification','fit');
[i, j] = find(X0==max(max(X0)));
q = [i, j]';

%% Define U and grad_U
X1 = size(I);
for i = 1 : size(I,1)
    for j = 1 : size(I,2)
        e = 0;
        for m = 1 : size(I,1)
            for n = 1 : size(I,2)
                r = sqrt((m-i)^2+(n-j)^2);
                if r ~= 0
                    e = e + I(m,n)/r;
                end 
            end
        end
        X1(i,j) = e;
    end
end
X1 = K * (X1 + I / InfD);
Uf = -(X1);

U = @(q) Uf(round(max(min(q(2),28),1)),...
            round(max(min(q(1),28),1)));
[px,py] = gradient(Uf);
grad_U = @(q)...
    [px(round(max(min(q(2),28),1)),...
        round(max(min(q(1),28),1))); ...
     py(round(max(min(q(2),28),1)),...
        round(max(min(q(1),28),1)))];

tr = [q'];
for i = 1 : NUM_SAMPLES
    % restart at half to avoid localization
    if i == round(NUM_SAMPLES / 2)
        q = [round(rand(1)*28) round(rand(1)*28)]';
    end
    q = hmc(U, grad_U, epsilon, L, q);
    q = [round(max(min(q(1),28),1)),...
        round(max(min(q(2),28),1))]';
    tr = [tr; q'];
end

x = tr(1:end-1,1); y = tr(1:end-1,2);
u = tr(2:end,1)-tr(1:end-1,1);
v = tr(2:end,2)-tr(1:end-1,2);

end

