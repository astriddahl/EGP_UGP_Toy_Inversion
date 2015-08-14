% Code to generate random features using fastfood
% (A Dahl July 2015)
% adapted for Matlab from python code: 
% """ Various basis function objects specialised for parameter learning.
% 
%     Authors:    Daniel Steinberg, Lachlan McCalman
%     Date:       8 May 2015
%     Institute:  NICTA
% 
% """


% Notes 
% currently assumes n>=d
% check that don't need lower bound >0 on [0,1) for draws of s_ii
% check OK to update PI within loop rather from I directly
% quick fix for univariate x - because Hadamard needs dim>=2?
% implement kernel estimate for cos sine expansion

%% 
clear;

% load input matrix x
load toyinvdata.mat;    % contains x (1000x1) and y
%load testdata.mat;      % contains x (100,1), [-pi,pi], K=rbf(ell=0.8) using covSEisoU.m from GPML toolbox

%% 
[m,d_init]=size(x);
nbases=500;                  % set dimension of finite feature expansion
sigma=0.9;                  %set scale for random gaussian matrix (corresponds to rbf kernel with sigma^2=0.9^2)

 
%% pad x until d is a power of 2, d>=1

% fix to pad x in the univariate case
if d_init==1
    xadd=zeros(length(x),1);
    x=[x,xadd];
    d_init=2;
end
% end fix    
    
l=ceil(log2(d_init));
d=2^l;                  % dim of expanded matrix X
k=ceil(nbases/d);
n=d*k;                  % total no features
d_add=d-d_init;
x_add=zeros(m,d_add);
X=[x,x_add];                % X expanded input matrix (m by d)
X_dash=X';


 
%% generate V
% for each input vector x (d x 1) generate V 1 to k, Vj is d x d, stack Vj
% to produce V (n x d)
% s_ii follow Steinberg and McCalman - sampling from chi-distn CDF
PI=eye(d);
VX=zeros(n,m);
for z = 1:k;
    B = diag(randsrc(d, 1, [1 -1]));
    G = diag(randn(d,1));
    perm=randperm(d);
    PI=PI(perm,:);
    S=diag(sqrt(2*(gammaincinv(rand(d,1),ceil(d/2)))))/norm(G);
    H=hadamard(d);
    V=(S*H*G*PI*H*B)/(sqrt(d)*sigma);       % first set of random features
    if z==1
         VX(z:d,:)=V*X_dash;                      %d x m feature input matrix
    else
        VX((z-1)*d+1:z*d,:)=V*X_dash;                %d x m feature input matrix
        %VX=vertcat(VX, VX_append);       %concat to n x m feature input matrix
    end
end

VX=VX';
PHI=(n^-0.5)*exp(1i*VX);              % return (m x n) feature matrix

%% export feature input matrix - unscaled - for egp/ugp
% phi_in defined as sigma*VX' (returns n x m (unscaled by sigma) feature input
% matrix)
phi_in=sigma*VX';
save('toyinvdata_phi_in.mat','phi_in');

%% cos-sin rather than complex exponential
PHI_cos=(n^-0.5)*cos(VX);
PHI_sin=(n^-0.5)*sin(VX);
PHI_cossin=cat(3,PHI_cos,PHI_sin);    %multidim array where i,j,1=cos; i,j,2=sin

save('toyinvdata_randfeatures.mat','PHI','PHI_cossin');
%% Approximate gram matrix using PHI

K_ff=zeros(m);
for i = 1:m
    for j = 1:m
        K_ff(i,j)=PHI(i,:)*PHI(j,:)';
    end
end

%% exact kernel - check of fastfood code/approximation to rbf
K_exactrbf=zeros(m);
for i = 1:m
    for j = 1:m
        K_exactrbf(i,j)=exp(-norm(x(i,:)-x(j,:))^2/(2*sigma^2));
    end
end

%%
save('toyinvdatakernels.mat','K_exactrbf','K_ff');