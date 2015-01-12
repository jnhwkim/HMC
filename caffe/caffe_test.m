%% Add matcaffe path 
addpath('/Users/Calvin/Github/caffe-master/matlab/caffe');

%% initialization
if caffe('is_initialized')
    caffe('reset');
end
samples = 49;
resample_tr = 8;
resample_te = 40;
use_gpu = 1;

%% Model Variation
variation = 'zj';
model_def_variation = 'zw';
model_variation = 'zwj';

%% Inputs
model_def_file = sprintf('hmc_deploy_%ds%dr%s.prototxt', ...
                    samples,resample_te,model_def_variation);
model_file = sprintf('.%ds%dr%s_iter_10000000.caffemodel', ...
                    samples,resample_tr,model_variation);
matcaffe_init(use_gpu, model_def_file, model_file);

%% Load test data
load(sprintf('../mat/%ds%dr%s_te.mat',samples,resample_te,variation));

%% Load MNIST data
addpath('../mnist');
y_te = loadMNISTLabels('t10k-labels-idx1-ubyte');

%% Forward
correct = 0;
count = 0;
error = zeros(1,10);
conflict = [];

for idx = 1 : size(S_te,1)/resample_te
    X = zeros(size(S_te(1,:), 2), ...
              size(S_te(1,:), 1), 1, resample_te);
    X(:,1,1,:) = S_te(resample_te*(idx-1)+1:resample_te*idx,:)';
    input_data = {single(uint8(X*255))*0.00390625};

    scores = caffe('forward', input_data);
    
    [~,maxlabel] = max(scores{1});
    if -mode(-maxlabel+1) ~= mode(maxlabel-1)
        conflict = [conflict; y_te(idx), squeeze(maxlabel-1)'];
    end
    if mode(maxlabel-1) == y_te(idx)
        correct = correct + 1;
    else
        if y_te(idx) == 0
            error(1,10) = error(1,10) + 1;
        else
            error(1,y_te(idx)) = error(1,y_te(idx)) + 1;
        end
        if visualization
            y_te(idx)
            LABEL = squeeze(maxlabel-1)';
            close all;
            f = figure(1);
            set(f, 'Position', [0 300 900 900]);
            for j = 1 : size(X,4)
                subplot(round(sqrt(resample_te)), ceil(sqrt(resample_te)), j);
                X1 = squeeze(X(:,1,1,j))';
                coord = 5;
                X2 = X1(1:size(X1,2)/coord*4);
                a = reshape(X2', [2 size(X2,2)/2 size(X2,1)]);
                z = X1(size(X1,2)/coord*4+1:end);
                coord = 4;
                x = squeeze(a(1,1:size(X2,2)/coord,:))'; 
                y = squeeze(a(2,1:size(X2,2)/coord,:))'; 
                u = squeeze(a(1,size(X2,2)/coord+1:end,:))'; 
                v = squeeze(a(2,size(X2,2)/coord+1:end,:))';
                %% Visualization
                imshow(ones(28), 'InitialMagnification','fit');
                hold on;
                quiver(x*28,y*28,(u-0.5)*56,(v-0.5)*56,0);
                title(LABEL(j), 'FontSize', 16);
            end
        end
    end
    count = count + 1;
end

% Display the result
fprintf('Accuracy = %.2f%%\n', correct / count * 100);
