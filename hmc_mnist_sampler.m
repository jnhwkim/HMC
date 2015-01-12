function [ x,y,u,v,z,Uf,px,py ] = hmc_mnist_sampler( X, q, samples, mask, filter )
%HMC_MNIST_SAMPLER Summary of this function goes here
%   Detailed explanation goes here

%% Debug messages?
DEBUG = false;

%% Hamilton Monte carlo sampling parameters
epsilon = .25;
L = 25;
K = .5;
InfD = .1;
I = reshape(X, [28 28]);
jump = true;

%% Initialize first gaze
if ~isequal(size(q), [2,1])
    if strcmp(q, 'max')
        f_mask = -fspecial('log', 7, 1);
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
if nargin < 5
    filter = coulomb_filter(size(I,1), size(I,2));
end
X1 = conv2(I, filter, 'same');
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
z = [U(q)];
jmp = [];
count = 0;

while size(tr,1) < samples+1 && count < samples * 5
    %% jump to the minima every quater.
    if mask && jump && 0 == mod(size(tr, 1), floor(samples / 4 + 1))
        [i, j] = find(Uf==min(min(Uf)));
        q = [j, i]';
        jmp = [jmp; size(tr,1)];
        if DEBUG 
            fprintf('jump to %d,%d at %d\n', ...
                q(1),q(2),size(tr,1));
        end
    end
    q = hmc(U, grad_U, epsilon, L, q);
    q = [round(max(min(q(1),28),1)),...
        round(max(min(q(2),28),1))]';
    count = count + 1;
    if ~isequal(tr(end,:), q') && (1 == size(tr,1) || ...
        1 < size(tr,1) && ~isequal(tr(end-1,:), q'))
        tr = [tr; q'];
        z = [z; U(q)];
        if mask
            % mask the middle of trajectory in addition to the destination.
            if 1 < size(tr,1) && ...
                0 ~= mod(size(tr, 1)-1, floor(samples / 4 + 1))
                q_trajectory = [tr(end,:)', round((tr(end-1,:)'+tr(end,:)')/2)]; 
            else
                q_trajectory = [tr(end,:)'];
            end
            for i = 1 : size(q_trajectory,2)
                q = q_trajectory(:,i);
                f_mask = hmc_mask(q,size(I),13,3);
                % Update the field.
                Z = sum(sum(Uf));
                minima = min(min(Uf));
                fraction = (Z + minima) / Z;
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
u(jmp) = 0; v(jmp) = 0;
z = z(1:end-1);

if DEBUG && size(x,1) < samples
    fprintf('Less sampled %d/%d\n', size(x,1), samples);
end

end

