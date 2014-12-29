addpath('mnist');
X_tr = loadMNISTImages('train-images-idx3-ubyte');
X_te = loadMNISTImages('t10k-images-idx3-ubyte');
y_tr = loadMNISTLabels('train-labels-idx1-ubyte');
y_te = loadMNISTLabels('t10k-labels-idx1-ubyte');
 
S_tr = hmc_mnist_bulk(X_tr, 60000);
S_te = hmc_mnist_bulk(X_te, 10000);

y_tr = y_tr(1:60000);
y_te = y_te(1:10000,:);

tr_label = '4S_tr';
te_label = '4S_te';

save(strcat(tr_label,'.mat'), 'S_tr');
save(strcat(te_label,'.mat'), 'S_te');

%% debug 
if false
   a = reshape(S_tr', [2 size(S_tr,2)/2 size(S_tr,1)]);
   x = squeeze(a(1,1:size(S_tr,2)/4,:))'; y = squeeze(a(2,1:size(S_tr,2)/4,:))'; 
   u = squeeze(a(1,size(S_tr,2)/4+1:end,:))'; v = squeeze(a(2,size(S_tr,2)/4+1:end,:))';
   %% Visualization
   close all;
   times = 4;
   idx = 5236;
   f = figure(1);
   set(f, 'Position', [0 300 300 300]);
   imshow(1-reshape(X_tr(:,ceil(idx/times)),28,28)/5, 'InitialMagnification','fit');
   hold on;
   quiver(x(idx,:)*28,y(idx,:)*28,(u(idx,:)-0.5)*56,(v(idx,:)-0.5)*56,0);
   y_tr(ceil(idx/times))
end

%% repeat sampling?
y_tr_rep = repmat(y_tr,1,4)';
y_te_rep = repmat(y_te,1,4)';
y_tr = y_tr_rep(:);
y_te = y_te_rep(:);

%% export as leveldb
export_leveldb(S_tr,y_tr,1,4*49,1,tr_label);
export_leveldb(S_te,y_te,1,4*49,1,te_label);
