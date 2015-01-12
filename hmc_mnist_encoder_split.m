%% Sampling for only test
half = 2;

%% Load MNIST data
addpath('mnist');
X_tr = loadMNISTImages('train-images-idx3-ubyte');
X_te = loadMNISTImages('t10k-images-idx3-ubyte');
y_tr = loadMNISTLabels('train-labels-idx1-ubyte');
y_te = loadMNISTLabels('t10k-labels-idx1-ubyte');

%% Split
if 1 == half
    X_tr = X_tr(:,1:35000);
elseif 2 == half
    X_tr = X_tr(:,35001:end);
end
 
%% HMC sampling
coord = 5;
samples = 49;
resample = 8;
variation = 'zo';

%% Set label names
tr_label = sprintf('%ds%dr%s_tr%d', samples, resample, variation, half);
te_label = sprintf('%ds%dr%s_te', samples, resample, variation);

%% Samplings and Save intermediate files
S_tr = hmc_mnist_bulk(X_tr, 60000, samples, resample, coord);
save(strcat('mat/',tr_label,'.mat'), 'S_tr');

%% For Test
if 1 ~= half
    S_te = hmc_mnist_bulk(X_te, 10000, samples, resample, coord);
    save(strcat('mat/',te_label,'.mat'), 'S_te');
end
