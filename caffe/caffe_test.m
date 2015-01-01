%% Add matcaffe path 
addpath('/Users/Calvin/Github/caffe-master/matlab/caffe');

%% initialization
if caffe('is_initialized')
    caffe('reset');
end
resample = 8;
use_gpu = 1;
model_def_file = sprintf('hmc_deploy_%ds.prototxt',resample);
model_file = sprintf('.%ds_ga_iter_10000000.caffemodel',resample);
matcaffe_init(use_gpu, model_def_file, model_file);

%% load test data
load(sprintf('../mat/%dS_te.mat',resample));

%% Load MNIST data
addpath('../mnist');
y_te = loadMNISTLabels('t10k-labels-idx1-ubyte');

%% forward
correct = 0;
count = 0;
for idx = 1 : 10000
    X = zeros(size(S_te(1,:), 2), ...
              size(S_te(1,:), 1), 1, resample);
    X(:,1,1,:) = S_te(resample*(idx-1)+1:resample*(idx-1)+resample,:)';
    input_data = {single(X)};

    scores = caffe('forward', input_data);
    [~,maxlabel] = max(scores{1});
    if mode(maxlabel-1) == y_te(idx)
        correct = correct + 1;
    end
    count = count + 1;
end

% Display the result
fprintf('Accuracy = %.2f%%\n', correct / count * 100);