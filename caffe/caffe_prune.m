%% Add matcaffe path 
addpath('/Users/Calvin/Github/caffe-master/matlab/caffe');

%% initialization
if caffe('is_initialized')
    caffe('reset');
end
coord = 4;
samples = 49;
resample = 8;
use_gpu = 1;
model_def_file = sprintf('hmc_deploy_%ds%dr.prototxt',samples,resample);
model_file = sprintf('.%ds%dr_ga_iter_10000000.caffemodel',samples,8);
matcaffe_init(use_gpu, model_def_file, model_file);

%% load test data
load(sprintf('../mat/%ds%dr_tr.mat',samples,resample));

%% Load MNIST data
addpath('../mnist');
y_tr = loadMNISTLabels('train-labels-idx1-ubyte');

%% forward
correct = 0;
count = 0;
error = zeros(1,10);
alive = [];

for idx = 1 : size(S_tr,1)/resample
    X = zeros(size(S_tr(1,:), 2), ...
              size(S_tr(1,:), 1), 1, resample);
    X(:,1,1,:) = S_tr(resample*(idx-1)+1:resample*(idx-1)+resample,:)';
    input_data = {single(uint8(X*255))*0.00390625};

    scores = caffe('forward', input_data);
    
    [~,maxlabel] = max(scores{1});
    a = sort(squeeze(scores{1}));
    
    alive = [alive; resample*(idx-1)+find(maxlabel-1 == y_tr(idx))];
    
    correct = correct + size(find(maxlabel-1 == y_tr(idx)),1);
    count = count + resample;
end

% Display the result
fprintf('Accuracy = %.4f%%\n', correct / count * 100);

%% Pruning
S_tr = S_tr(alive,:);
y_tr_rep = repmat(y_tr,1,resample)';
y_tr = y_tr_rep(:);
clear('y_tr_rep')
y_tr = y_tr(alive,:);

addpath('..');
addpath('../protobuf');
tp_label = sprintf('%ds%dr_tp', samples, resample);
save(sprintf('../mat/%s.mat',tp_label), 'S_tr');
export_leveldb(S_tr,y_tr,1,coord*samples,1,tp_label);
