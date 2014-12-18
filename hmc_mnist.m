%% Example for two-dimensional problem

set(0,'DefaultAxesFontName', 'Palatino')
set(0,'DefaultTextFontname', 'Palatino')

%% Load a sample digit
addpath('mnist');
[X,y] = loadadigit('train',round(rand(1)*100+1));
X = reshape(X, [28 28]);

q = [14 0]';
epsilon = .25;
L = 10;
K = .5;
InfD = .1;
NUM_SAMPLES = 1000;

%% Define U and grad_U
X1 = size(X);
for i = 1 : size(X,1)
    for j = 1 : size(X,2)
        e = 0;
        for m = 1 : size(X,1)
            for n = 1 : size(X,2)
                r = sqrt((m-i)^2+(n-j)^2);
                if r ~= 0
                    %e = e + [(m-i),(n-j)]*X(m,n)/(r^2);
                    e = e + X(m,n)/r;
                end 
            end
        end
        X1(i,j) = e;
    end
end
X1 = K * (X1 + X / InfD);

%Z = sum(sum(exp(-X1.^2/2)));
%Uf = X1.^2/2 + log(Z);
Uf = -(X1);

U = @(q) Uf(round(max(min(q(2),28),1)),...
            round(max(min(q(1),28),1)));
[px,py] = gradient(Uf);
grad_U = @(q)...
    [px(round(max(min(q(2),28),1)),...
        round(max(min(q(1),28),1))); ...
     py(round(max(min(q(2),28),1)),...
        round(max(min(q(1),28),1)))];

tr = [q'];
for i = 1 : NUM_SAMPLES
    % restart at half to avoid localization
    if i == round(NUM_SAMPLES / 2)
        q = [14 28]';
    end
    q = hmc(U, grad_U, epsilon, L, q);
    q = [round(max(min(q(1),28),1)),...
        round(max(min(q(2),28),1))]';
    tr = [tr; q'];
end

x = tr(1:end-1,1); y = tr(1:end-1,2);
u = tr(2:end,1)-tr(1:end-1,1); 
v = tr(2:end,2)-tr(1:end-1,2);

close all;
f1 = figure(1);

imshow(1-X/2, 'InitialMagnification','fit');
set(f1, 'Position', [0 0 300 300]);
hold on;
quiver(x,y,u,v,0);

xlabel('x', 'FontSize', 16);
ylabel('y', 'FontSize', 16);
set(gca, 'FontSize', 13);

% f2 = figure(2);
% set(f2, 'Position', [0 300 300 300]);
% scatter(tr(:,1),28-tr(:,2));
% axis([1 28 1 28]);

f2 = figure(2);
set(f2, 'Position', [0 300 300 300]);
imshow(X1, jet(18), 'InitialMagnification','fit');
% scatter(tr(:,1),28-tr(:,2));
% axis([1 28 1 28]);

f3 = figure(3);
set(f3, 'Position', [0 600 300 300]);
quiver(flipud(px),flipud(-py));
axis([1 28 1 28]);