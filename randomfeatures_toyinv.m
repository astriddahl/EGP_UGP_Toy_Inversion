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


%% Notes 
% currently assumes n>d ie feature dimension > input data - need to
% % update to allow for n<d
% check that don't need lower bound >0 on [0,1) for draws of s_ii
% check OK to update PI within loop rather from I directly
% quick fix for univariate x - because Hadamard needs dim>=2?

%% 
clear;

% load input matrix x
load toyinvdata.mat;    % contains x (1000x1)

[m,d_init]=size(x);
nbases=6;                  % set dimension of finite feature expansion
sigma=1;                  %set scale for random gaussian matrix

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


%% 
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
F=(n^-0.5)*exp(1i*VX);              % return (m x n) feature matrix

save('toyinvdata_randfeatures.mat','F');

