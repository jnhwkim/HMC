addpath('mnist');
X_tr = loadMNISTImages('train-images-idx3-ubyte');
X_te = loadMNISTImages('t10k-images-idx3-ubyte');
y_tr = loadMNISTLabels('train-labels-idx1-ubyte');
y_te = loadMNISTLabels('t10k-labels-idx1-ubyte');
 
S_tr = hmc_mnist_bulk(X_tr, 60000);
S_te = hmc_mnist_bulk(X_te, 10000);

y_tr = y_tr(1:60000);
y_te = y_te(1:10000,:);

save('S_tr.mat', 'S_tr');
save('S_te.mat', 'S_te');

if true
   a = reshape(S_tr(:,1:784)', [2 784/2 size(S_tr,1)]);
   x = squeeze(a(1,1:196,:))'; y = squeeze(a(2,1:196,:))'; 
   u = squeeze(a(1,197:end,:))'; v = squeeze(a(2,197:end,:))';
   %% Visualization
%    close all;
%    idx = 66;
%    f = figure(1);
%    set(f, 'Position', [0 300 300 300]);
%    imshow(ones(28), 'InitialMagnification','fit');
%    hold on;
%    quiver(x(idx,:)*28,y(idx,:)*28,(u(idx,:)-0.5)*56,(v(idx,:)-0.5)*56,0);
end

export_leveldb(S_tr,y_tr,28,28*2,1,'S_tr');
export_leveldb(S_te,y_te,28,28*2,1,'S_te');

%%
export_leveldb(S_tr(:,1:784),y_tr,28,28,1,'S_tr');
export_leveldb(S_te(:,1:784),y_tr,28,28,1,'S_te');