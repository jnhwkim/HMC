function [ x,y,u,v,Uf,px,py ] = hmc_mnist_sampler( X, q, samples, mask )
%HMC_MNIST_SAMPLER Summary of this function goes here
%   Detailed explanation goes here

% Hamilton Monte carlo sampling parameters
epsilon = .25;
L = 25;
K = .5;
InfD = .1;
I = reshape(X, [28 28]);

%% Initialize first gaze
if ~isequal(size(q), [2,1])
    if strcmp(q, 'max')
        f_size = 7;
        f_mask = fspecial('gaussian', [f_size f_size], 0.5) - ...
                 ones(f_size,f_size) / f_size^2;
        X0 = conv2(I, f_mask, 'same');
        %f0 = figure(4);
        %set(f0, 'Position', [300 0 300 300]);
        %imshow(X0*30, jet(36), 'InitialMagnification','fit');
        % add some randomness
        X0 = X0 + rand(size(X0))*(max(X0(:))-median(X0(:))) / 2;
        % avoid center
        X0 = X0 - fspecial('gaussian', size(X0), 1);
        [i, j] = find(X0==max(max(X0)));
        q = [j, i]';
    end
end

%% Define U and grad_U
X1 = size(I);
for i = 1 : size(I,1)
    for j = 1 : size(I,2)
        e = 0;
        for m = 1 : size(I,1)
            for n = 1 : size(I,2)
                r = sqrt((m-i)^2+(n-j)^2);
                if r ~= 0
                    e = e + I(m,n)/r;
                end 
            end
        end
        X1(i,j) = e;
    end
end

X1 = K * (X1 + I / InfD);

Uf = -(X1);
[px,py] = gradient(Uf);

U = @(q) Uf(round(max(min(q(2),28),1)),...
            round(max(min(q(1),28),1)));
grad_U = @(q)...
    [px(round(max(min(q(2),28),1)),...
        round(max(min(q(1),28),1))); ...
     py(round(max(min(q(2),28),1)),...
        round(max(min(q(1),28),1)))];

tr = [q'];
count = 0;

while size(tr,1) < samples && count < samples * 10
    q = hmc(U, grad_U, epsilon, L, q);
    q = [round(max(min(q(1),28),1)),...
        round(max(min(q(2),28),1))]';
    count = count + 1;
    if ~isequal(tr(end,:), q') && (1 == size(tr,1) || ...
        1 < size(tr,1) && ~isequal(tr(end-1,:), q'))
        tr = [tr; q'];
        if mask
            % mask the middle of trajectory in addition to the destination.
            if 1 < size(tr,1)
                q_trajectory = [tr(end,:)', round((tr(end-1,:)'+tr(end,:)')/2)]; 
            else
                q_trajectory = [tr(end,:)'];
            end
            for i = 1 : size(q_trajectory,2)
                q = q_trajectory(:,i);
                f_size = 13;
                f_mask = zeros(size(I,1)+f_size-1, size(I,2)+f_size-1);
                offset = floor(f_size/2);
                f_mask(q(2):q(2)+f_size-1,q(1):q(1)+f_size-1) = ...
                    fspecial('gaussian', [f_size f_size], 3);
                f_mask = f_mask(offset+1:end-offset,offset+1:end-offset);
                % Update the field.
                minima = min(min(Uf));
                fraction = (sum(sum(Uf)) + minima) / sum(sum(Uf));
                Uf = Uf * fraction - minima * f_mask;
                [px,py] = gradient(Uf);
                U = @(q) Uf(round(max(min(q(2),28),1)),...
                    round(max(min(q(1),28),1)));
                grad_U = @(q)...
                    [px(round(max(min(q(2),28),1)),...
                        round(max(min(q(1),28),1))); ...
                     py(round(max(min(q(2),28),1)),...
                        round(max(min(q(1),28),1)))];
            end
        end
    end
end

x = tr(1:end-1,1); y = tr(1:end-1,2);
u = tr(2:end,1)-tr(1:end-1,1);
v = tr(2:end,2)-tr(1:end-1,2);

end

