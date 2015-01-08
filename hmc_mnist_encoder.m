%% Load MNIST data
addpath('mnist');
X_tr = loadMNISTImages('train-images-idx3-ubyte');
X_te = loadMNISTImages('t10k-images-idx3-ubyte');
y_tr = loadMNISTLabels('train-labels-idx1-ubyte');
y_te = loadMNISTLabels('t10k-labels-idx1-ubyte');
 
%% HMC sampling
coord = 5;
samples = 49;
resample = 8;
variation = 'z';

%% Set label names
tr_label = sprintf('%ds%dr%s_tr', samples, resample, variation);
te_label = sprintf('%ds%dr%s_te', samples, resample, variation);

%% Samplings and Save intermediate files
S_tr = hmc_mnist_bulk(X_tr, 60000, samples, resample, coord);
save(strcat('mat/',tr_label,'.mat'), 'S_tr');

%% For Test
S_te = hmc_mnist_bulk(X_te, 10000, samples, resample, coord);
save(strcat('mat/',te_label,'.mat'), 'S_te');

%% debug 
if false
   a = reshape(S_tr', [2 size(S_tr,2)/2 size(S_tr,1)]);
   x = squeeze(a(1,1:size(S_tr,2)/coord,:))'; y = squeeze(a(2,1:size(S_tr,2)/coord,:))'; 
   u = squeeze(a(1,size(S_tr,2)/coord+1:end,:))'; v = squeeze(a(2,size(S_tr,2)/coord+1:end,:))';
   %% Visualization
   close all;
   resample = 4;
   idx = 1;
   f = figure(1);
   set(f, 'Position', [0 300 300 300]);
   imshow(1-reshape(X_tr(:,ceil(idx/times)),28,28)/5, 'InitialMagnification','fit');
   hold on;
   quiver(x(idx,:)*28,y(idx,:)*28,(u(idx,:)-0.5)*56,(v(idx,:)-0.5)*56,0);
   y_tr(ceil(idx/times))
end

%% repeat sampling?
y_tr_rep = repmat(y_tr,1,resample)';
y_te_rep = repmat(y_te,1,resample)';
y_tr = y_tr_rep(:);
y_te = y_te_rep(:);
clear('y_tr_rep')
clear('y_te_rep')

%% export as leveldb
export_leveldb(S_tr,y_tr,1,coord*samples,1,tr_label);
export_leveldb(S_te,y_te,1,coord*samples,1,te_label);
