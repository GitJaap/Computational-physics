clc; clear all; close all;
% Numerical params
h = 1/2000;
a = 1/2000;
x = [0:a:1]';
N = length(x);
T = 200;

% Physical params
V = zeros(size(x));

% Initial cond
mu = 0.5;
sigma = 0.05;
k0 = 10;
psi0= 1/sqrt(sigma*sqrt(2*pi)) * exp(-(x-mu).^2/(4*sigma^2) + j*k0*x);

% Program
Ha = 1/a^2/2 * spdiags([-ones(N,1) 2*ones(N,1) -ones(N,1)],[-1 0 1],N,N) + spdiags(V,0,N,N);

M = i/h * speye(N) - Ha/2;
A = i/h * speye(N) + Ha/2;

psi = zeros(N,T);
psi(:,1) = psi0;

for i = 2:T
    i
    psi(:,i) = M\(A*psi(:,i-1));
    figure(1)
    plot(x,abs(psi(:,i)).^2);
    axis([0 1 0 8])
    pause(0.1);
end

