function add_intensity
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

%% Load MNIST data
addpath('mnist');
X_tr = loadMNISTImages('train-images-idx3-ubyte');
X_te = loadMNISTImages('t10k-images-idx3-ubyte');

%% Load target matrix
samples = 49;
resample = 8;
variation = '';
%load(sprintf('mat/%ds%dr%s_tr.mat', samples, resample, variation));
load(sprintf('mat/%ds%dr%s_te.mat', samples, resample, variation));

%S_tr = do_add_intensity(X_tr, S_tr, resample); %#ok<*NASGU,*NODEF>
S_te = do_add_intensity(X_te, S_te, resample);

variation = 'ii';
%save(sprintf('mat/%ds%dr%s_tr.mat', samples, resample, variation), 'S_tr');
save(sprintf('mat/%ds%dr%s_te.mat', samples, resample, variation), 'S_te');

end

function X1 = do_add_intensity( I, X, resample )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

%% Precondition
assert(0 == mod(size(X, 1), resample));
assert(0 == mod(size(X, 2), 4));

height = 28;
width = 28;
X1 = zeros(size(X, 1), size(X, 2) + size(X, 2) / 4);
X1(:,1:size(X, 2)) = X;

for i = 1 : size(X, 1)
    coords = reshape(X(i,1:size(X,2)/2), [2 size(X,2)/4]);
    x = coords(1,:)' * width;
    y = coords(2,:)' * height;
    z = uint16((x-1)*height+y);
    z = z(z>0);
    c = I(z, ceil(i / resample));
    X1(i, size(X, 2) + 1 : size(X, 2) + size(c, 1)) = c';
end

end