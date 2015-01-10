%% Load MNIST data
addpath('mnist');
X_tr = loadMNISTImages('train-images-idx3-ubyte')';
X_te = loadMNISTImages('t10k-images-idx3-ubyte')';
y_tr = loadMNISTLabels('train-labels-idx1-ubyte');
y_te = loadMNISTLabels('t10k-labels-idx1-ubyte');

%% Variables
samples = size(X_tr,2)/4;
coord = 5;
height = 28;
width = 28;

%% Set label names
tr_label = sprintf('%ds%dc_tr', samples, coord);
te_label = sprintf('%ds%dc_te', samples, coord);

%% Sampling
S_tr = coord_sampler(X_tr, height, width, samples);
save(sprintf('mat/%s.mat', tr_label), 'S_tr');
S_te = coord_sampler(X_te, height, width, samples);
save(sprintf('mat/%s.mat', te_label), 'S_te');

%% export as leveldb
export_leveldb(S_tr,y_tr,1,coord*samples,1,tr_label);
export_leveldb(S_te,y_te,1,coord*samples,1,te_label);