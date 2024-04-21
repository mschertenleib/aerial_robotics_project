close all;
clear;
clc;

x_min = -3;
x_max = 3;
N = 500;
[x, y] = meshgrid(linspace(x_min, x_max, N), linspace(x_min, x_max, N));

p1x = 1.0;
p1y = 0;
p2x = 0;
p2y = 0;
p3x = -1.0;
p3y = 0;
f = @(x, y, px, py) max(1 - sqrt((x - px).^2 + (y - py).^2), 0).^2;
f1 = @(x, y) f(x, y, p1x, p1y);
f2 = @(x, y) f(x, y, p2x, p2y);
f3 = @(x, y) f(x, y, p3x, p3y);

% % z(p) = sum(a_j*f_j(p))
% % z(p_i) = sum(a_j*f_j(p_i)) = f_i(p_i)
% 
% F = [[f1(p1x,p1y), f2(p1x,p1y), f3(p1x,p1y)];
%     [f1(p2x,p2y), f2(p2x,p2y), f3(p2x,p2y)];
%     [f1(p3x,p3y), f2(p3x,p3y), f3(p3x,p3y)]];
% fv = [f1(p1x,p1y), f2(p2x,p2y), f3(p3x,p3y)]';
% a = F \ fv;
% z = a(1)*f1(x,y)+a(2)*f2(x,y)+a(3)*f3(x,y);

f = zeros(N, N, 3);
f(:,:,1) = f1(x,y);
f(:,:,2) = f2(x,y);
f(:,:,3) = f3(x,y);
z = smooth_max(f, 1);

figure;
surf(x, y, z, 'EdgeColor', 'none');
hold on;
contour3(x, y, z, 20, '-k');
hold off;
colormap('jet');
shading('interp');


function y = smooth_max(x, alpha)
exp_norms = exp(alpha .* x);
y = sum(x .* exp_norms, 3) ./ sum(exp_norms, 3);
end
