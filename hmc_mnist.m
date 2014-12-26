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

tic;
[x,y,u,v,X1,px,py] = hmc_mnist_sampler(X, [0;0], 100);
toc;

f1 = figure(1);

imshow(1-X/2, 'InitialMagnification','fit');
set(f1, 'Position', [0 0 300 300]);
hold on;
quiver(x,y,u,v,0);

xlabel('x', 'FontSize', 16);
ylabel('y', 'FontSize', 16);
set(gca, 'FontSize', 13);

% f2 = figure(2);
% set(f2, 'Position', [0 300 300 300]);
% scatter(tr(:,1),28-tr(:,2));
% axis([1 28 1 28]);

f2 = figure(2);
set(f2, 'Position', [0 300 300 300]);
imshow(X1, jet(18), 'InitialMagnification','fit');
% scatter(tr(:,1),28-tr(:,2));
% axis([1 28 1 28]);

f3 = figure(3);
set(f3, 'Position', [0 600 300 300]);
quiver(flipud(px),flipud(-py));
axis([1 28 1 28]);