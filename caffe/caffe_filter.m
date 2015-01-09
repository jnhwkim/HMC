%function [ output_args ] = caffe_filter( input_args )
%CAFFE_FILTER Summary of this function goes here
%   Detailed explanation goes here

%% Add matcaffe path 
addpath('/Users/Calvin/Github/caffe-master/matlab/caffe');

%% Load MNIST data
addpath('../mnist');
X_tr = loadMNISTImages('train-images-idx3-ubyte');
X_te = loadMNISTImages('t10k-images-idx3-ubyte');
y_tr = loadMNISTLabels('train-labels-idx1-ubyte');
y_te = loadMNISTLabels('t10k-labels-idx1-ubyte');

%% initialization
if caffe('is_initialized')
    caffe('reset');
end

use_gpu = 1;
model_def_file = sprintf('lenet_deploy.prototxt');
model_file = sprintf('.lenet_iter_10000.caffemodel');
matcaffe_init(use_gpu, model_def_file, model_file);

for i = 1 : 10
    % Caffe uses width-first dims but train was height-first.
    X = X_te(:,i)';
    input_data = {single(uint8(X*255))*0.00390625};
    scores = caffe('forward', input_data);
    [~,maxLabel] = max(scores{1});
    assert(maxLabel-1 == y_te(i)); % Bear in mind that accuracy is .99.
end

net = caffe('get_weights');
weights = net.weights;
filters = weights{1};

figure(1);
for i  = 1 : size(filters, 4)
    subplot(5,4,i);
    imshow(squeeze(filters(:,:,:,i))*256, jet(256));
end


%end