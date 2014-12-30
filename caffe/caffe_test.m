%% Add matcaffe path 
addpath('/Users/Calvin/Github/caffe-master/matlab/caffe');

%% initialization
use_gpu = 1;
model_def_file = 'hmc_deploy.prototxt';
model_file = '.4s_ga_iter_100000.caffemodel';
matcaffe_init(use_gpu, model_def_file, model_file);

%% load test data
load('../4S_te.mat');

%% Load MNIST data
addpath('../mnist');
y_te = loadMNISTLabels('t10k-labels-idx1-ubyte');

%% forward
correct = 0;
count = 0;
for idx = 1 : 10000
    samples = 4;
    X = zeros(size(S_te(1,:), 2), ...
              size(S_te(1,:), 1), 1, samples);
    X(:,1,1,:) = S_te(samples*(idx-1)+1:samples*(idx-1)+samples,:)';
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