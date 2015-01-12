
%% HMC sampling
coord = 5;
samples = 49;
resample = 8;
variation = 'zmj';

%% Set label names
tr_label = sprintf('%ds%dr%s_tr', samples, resample, variation)
te_label = sprintf('%ds%dr%s_te', samples, resample, variation)

%% Load matrices.
load(fprintf('mat/%ds%dr%s_tr%d.mat', samples, resample, variation, 1));
S1 = S_tr;
load(fprintf('mat/%ds%dr%s_tr%d.mat', samples, resample, variation, 2));
S2 = S_tr;

%% Merge
S_tr = [S1; S2];
%save(fprintf('mat/%ds%dr%s_tr.mat', samples, resample, variation), 'S_tr');

%% Load S_te
load(fprintf('mat/%ds%dr%s_te.mat', samples, resample, variation));

%% Load MNIST data
addpath('mnist');
y_tr = loadMNISTLabels('train-labels-idx1-ubyte');
y_te = loadMNISTLabels('t10k-labels-idx1-ubyte');

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
