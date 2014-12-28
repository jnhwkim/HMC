%% Na?ve approach to sample MNIST digits using Hamilton Monte Carlo.
%% Jin-Hwa Kim (jnhwkim@snu.ac.kr)
%% Updated on Dec 26 2014

close all;

set(0,'DefaultAxesFontName', 'Palatino')
set(0,'DefaultTextFontname', 'Palatino')

%% Load a sample digit
addpath('mnist');
[X,y] = loadadigit('train',1);%round(rand(1)*100+1));
X = reshape(X, [28 28]);
samples = 28^2/4/4; % 49

%% Init with Zero.
tic;
[x,y,u,v,Uf,px,py] = hmc_mnist_sampler(X, [0;0], samples, false);
toc;

%% Display the result as 3 figures.
[f1,f2,f3] = hmc_mnist_disp(X,x,y,u,v,-Uf,px,py,0);

%% Modified approach
tic;
[x,y,u,v,Uf,px,py] = hmc_mnist_sampler(X, 'max', samples, true);
toc;

%% Display the result as 3 figures.
[f4,f5,f6] = hmc_mnist_disp(X,x,y,u,v,-Uf,px,py,300);
