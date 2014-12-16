%% Example for two-dimensional problem

set(0,'DefaultAxesFontName', 'Palatino')
set(0,'DefaultTextFontname', 'Palatino')

%% Load a sample digit
addpath('mnist');
[X,y] = loadadigit('train',1);
X = reshape(X, [28 28]); % 5

q = [14 14]';
epsilon = 0.2;
L = 25;

%% Define U and grad_U
%filter = fspecial('gaussian', 7, 5);
%Xf = imfilter(X, filter, 'replicate');

%X = X + 200 * fspecial('gaussian', 28, 8);

X(1,:) = 0; X(end,:) = 0;
X(:,1) = 0; X(:,end) = 0;

Z = sum(sum(exp(-X.^2/2)));
Uf = X.^2/2 - log(Z);

U = @(q) Uf(round(max(min(q(1),28),1)),...
            round(max(min(q(2),28),1)));
[px,py] = gradient(Uf);
grad_U = @(q)...
    [px(round(max(min(q(1),28),1)),...
        round(max(min(q(2),28),1))); ...
     py(round(max(min(q(1),28),1)),...
        round(max(min(q(2),28),1)))];

tr = [q'];
for i = 1 : 100
    q = hmc(U, grad_U, epsilon, L, q);
    q = [round(max(min(q(1),28),1)),...
        round(max(min(q(2),28),1))]';
    tr = [tr; q'];
    
end

x = tr(1:end-1,1); y = tr(1:end-1,2);
u = tr(2:end,1)-tr(1:end-1,1); 
v = tr(2:end,2)-tr(1:end-1,2);

close all;
f = figure(1);

imshow(1-X);
set(f, 'Position', [100 300 500 500]);
hold on;

quiver(x,y,u,v,0);

xlabel('x', 'FontSize', 16);
ylabel('y', 'FontSize', 16);
set(gca, 'FontSize', 13);

figure(2);
scatter(x,-y);