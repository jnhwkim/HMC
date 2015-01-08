%% Add matcaffe path 
addpath('/Users/Calvin/Github/caffe-master/matlab/caffe');

%% initialization
if caffe('is_initialized')
    caffe('reset');
end
samples = 49;
resample = 8;
use_gpu = 1;
variation = 'z';
model_def_file = sprintf('hmc_deploy_%ds%dr%s.prototxt', ...
                    samples,resample,variation);
model_file = sprintf('.%ds%dr%s_iter_10000000.caffemodel', ...
                    samples,8,variation);
matcaffe_init(use_gpu, model_def_file, model_file);

%% load test data
load(sprintf('../mat/%ds%dr%s_te.mat',samples,resample,variation));

%% Load MNIST data
addpath('../mnist');
y_te = loadMNISTLabels('t10k-labels-idx1-ubyte');

%% forward
correct = 0;
count = 0;
error = zeros(1,10);

for idx = 1 : size(S_te,1)/resample
    X = zeros(size(S_te(1,:), 2), ...
              size(S_te(1,:), 1), 1, resample);
    X(:,1,1,:) = S_te(resample*(idx-1)+1:resample*(idx-1)+resample,:)';
    input_data = {single(uint8(X*255))*0.00390625};

    scores = caffe('forward', input_data);
    
    [~,maxlabel] = max(scores{1});
    if mode(maxlabel-1) == y_te(idx)
        correct = correct + 1;
    else 
        if y_te(idx) == 0
            error(1,10) = error(1,10) + 1;
        else
            error(1,y_te(idx)) = error(1,y_te(idx)) + 1;
        end
    end
    count = count + 1;
end

% Display the result
fprintf('Accuracy = %.2f%%\n', correct / count * 100);
