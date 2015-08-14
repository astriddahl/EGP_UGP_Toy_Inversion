%% generate data as in toy inversion example (steinberg and bonilla)
% code by adahl July 2015

% Notes 
% per paper, n=1000, x evenly spread over [-2pi,2pi], f~N(0,K),
% Matern kernel(5/2) (ie d=5), amp=0.8,length=0.6 
% requires GPML toolbox
% http://www.gaussianprocess.org/gpml/code/matlab/doc/

% alternate use for RBF kernel: implemented for sigma=0.9

clear;
n=1000;
x=linspace(-2*pi,2*pi,n)';
ell=0.9
sf=0.6

addpath(genpath('C:\Program Files\MATLAB\R2014b\astridlib')); %requires minfunc, opt toolbox and (for toyinvdata generation) gpmltoolbox
%matern kernel
%covfunc = {@covMaterniso, 5}; hyp.cov = log([ell; sf]); % Matern class d=5
% rbf kernel
covfunc = {@covSEisoU}; hyp.cov = log([ell]); % rbf kernel with sigma^2=0.9^2.
K = feval(covfunc{:}, hyp.cov, x);

%sample f from GP (code from Mathmonk19.4)
u=randn(n,1);   %std normal (0,1)
[A,S,B]=svd(K); % factor K=ASB'
f=A*sqrt(S)*u;  % f~N(0,K)

plot(x, f, '+')

%% generate y=g(f)+e, e~N(0,0.2^2)
% sige=0.2
% e=randn(n,1)*sige;

%for debugging generate more variable ydata:
sige=2;
e=randn(n,1)*sige;

% nonlinear functions g(f)
% toyinv functions tested are f, f+f^2+f^3, exp(f), sin(f), tanh(2f)

nnlf=5        % no. alternative nonlinear functions
Y=zeros(n,nnlf);
for i=1:nnlf
    nlf=i;
    switch nlf
        case 1; g=@(f) f;
        case 2; g=@(f) f+f.^2+f.^3;
        case 3; g=@(f) exp(f);
        case 4; g=@(f) sin(f);
        case 5; g=@(f) tanh(f);
    end
    Y(:,i)=g(f)+e;
end

save('toyinvdata.mat','x','Y','f');
%% end file

 