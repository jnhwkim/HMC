%% Example for two-dimensional problem

set(0,'DefaultAxesFontName', 'Palatino')
set(0,'DefaultTextFontname', 'Palatino')

q = [-3 -3]';
epsilon = 0.25;
L = 25;

S = [1 0; 0 1];
U = @(q) q' * (S \ q) / 2;
grad_U = @(q) (q \ inv(S))';

tr = [q'];
for i = 1 : 300
    q = hmc(U, grad_U, epsilon, L, q);
    tr = [tr; q'];
end

w = 3;
h = 3;

x = tr(1:end-1,1); y = tr(1:end-1,2);
u = tr(2:end,1)-tr(1:end-1,1); 
v = tr(2:end,2)-tr(1:end-1,2);

close all;
f = figure(1);
set(f, 'Position', [100 300 500 500]);
quiver(x,y,u,v,0);
axis([-w,w,-h,h]);
xlabel('x', 'FontSize', 16);
ylabel('y', 'FontSize', 16);
set(gca, 'XTick', -w:w, 'XTickLabel', -w:w, 'FontSize', 13);