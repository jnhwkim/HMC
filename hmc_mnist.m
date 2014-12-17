%% Example for two-dimensional problem

set(0,'DefaultAxesFontName', 'Palatino')
set(0,'DefaultTextFontname', 'Palatino')

%% Load a sample digit
addpath('mnist');
[X,y] = loadadigit('train',1);
X = reshape(X, [28 28]); % 5

q = [1 1]';
epsilon = 1;
L = 1;

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

Z = sum(sum(X1));
Uf = X1 / Z;

U = @(q) Uf(round(max(min(q(1),28),1)),...
            round(max(min(q(2),28),1)));
[px,py] = gradient(Uf);
grad_U = @(q)...
    [px(round(max(min(q(1),28),1)),...
        round(max(min(q(2),28),1))); ...
     py(round(max(min(q(1),28),1)),...
        round(max(min(q(2),28),1)))];

tr = [q'];
for i = 1 : 1500
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

imshow(1-X/2);
set(f, 'Position', [100 300 500 500]);
hold on;

quiver(x,y,u,v,0);
%quiver(px,py);

xlabel('x', 'FontSize', 16);
ylabel('y', 'FontSize', 16);
set(gca, 'FontSize', 13);

figure(2);
scatter(x,-y);