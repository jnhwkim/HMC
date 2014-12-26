function [ f1,f2,f3 ] = hmc_mnist_disp( X,x,y,u,v,X1,px,py,xpos )
%HMC_MNIST_DISP Summary of this function goes here
%   Detailed explanation goes here

f1 = figure;
imshow(1-X/2, 'InitialMagnification','fit');
set(f1, 'Position', [xpos 0 300 300]);
hold on;
quiver(x,y,u,v,0);

xlabel('x', 'FontSize', 16);
ylabel('y', 'FontSize', 16);
set(gca, 'FontSize', 13);

% f2 = figure(2);
% set(f2, 'Position', [0 300 300 300]);
% scatter(tr(:,1),28-tr(:,2));
% axis([1 28 1 28]);

f2 = figure;
set(f2, 'Position', [xpos 300 300 300]);
imshow(X1, jet(18), 'InitialMagnification','fit');
% scatter(tr(:,1),28-tr(:,2));
% axis([1 28 1 28]);

f3 = figure;
set(f3, 'Position', [xpos 600 300 300]);
quiver(flipud(px),flipud(-py));
axis([1 28 1 28]);

end

