% file to automate prediction tasks - calls multitask_egp_ugp.m
% A Dahl 18 Aug 2015
% prediction of f and y for five g(f) models (nlf=1:5) where
%     f_qn=w_q*phi_n and y_n=g(f_n)+e
%     y_n~N(A*M*phi_n,DELTA+sumq(a_q*phi_n'*C_q*phi_n*a_q')  - EQN(28)
%     thus y_n_hat=A*M*phi_n   from EQN(28)
%     
%     f_qn_hat=m_q*phi_n
%     P(f_qn)=P(w_q*phi_n) where phi_n is treated as a constant,
%                          and w_q~N(m_q,C_q)
%     thus P(f_qn|phi_n,M,C)~N(phi_n'm,phi_n'Cphi_n)

%Notes:
% - currently written for univariate f

clear;
% nonlinear models to be tested (out of nlf=1:5)
nlfvec=[1,2,4,5];

% number of folds - cross validation
k=5;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% code for toyinvdata: Y matrix contains 5 alternative values for Y nlf
% toyinvdata and phi_in parameter values:
    %true RBF and random features sigma=0.9   
    % true sigma_e for all y=g(f)+e sige=0.2
    % true sigma_f for f=phiw sigma_f=1 

    load 'toyinvdata.mat';      % true f also included for toyinvdata.mat
    load toyinvdata_phi_in.mat;   % contains "phi_in" random matrix DxN - input to generate PHI

    % code to break up toyinvdata Y and choose which (Y1 to Y5) - works for
    % P=1
    Yall=Y;
    datain=[f Yall phi_in'];
% end of code for toyinvdata
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%separate data into k-folds 

%     randomly subset k/N rows(Y_allcases,f) and associated cols(phi_in) 
Nall=length(datain);
rowperm=randperm(Nall);        % row vector of row/col IDs
%%
data=zeros(Nall,size(datain,2));
    for r=1:Nall
    data(r,:)=(data(rowperm(r),:));
    end
%%
if mod(Nall,k)~=0
   display('warning: number of folds (k) is not a divisor of N. This prog is not currently set up to partition with non-divisors.'); 
   pause;
end

% loop across folds
xvresults={k,length(nlfvec)};
save 'xvresults.mat' xvresults;
for c=1:k
    
    % Partition data into test sets and train sets
    idstart=(c-1)*(Nall/k)+1;
    idend=c*(Nall/k);
    
    % create vars from fold data (randomly reordered)
    f=data(:,1);
    Yall=data(1:end,size(f,2)+1:size(Yall,2)+1);
    phi_in=data(1:end,size(f,2)+size(Yall,2)+1:end);
    
    % create test sets
    ftest=f(idstart:idend);
    Yalltest=Yall(idstart:idend,1:end);
    phi_intest=phi_in(idstart:idend,1:end);
    phi_intest=phi_intest';
    
    % create train sets
    if c==1
        f=f(idend+1:end);
        Yall=Yall(idend+1:end,1:end);
        phi_in=phi_in(idend+1:end,1:end);
    elseif c<k && c>1
        f1=f(1:idstart-1);
        f2=f(idend+1:end);
        f=[f1;f2];
        yk1=Yall(1:idstart-1,1:end);
        yk2=Yall(idend+1:end,1:end);
        Yall=[yk1;yk2];
        pk1=phi_in(1:idstart-1,1:end);
        pk2=phi_in(idend+1:end,1:end);
        phi_in=[pk1;pk2];
    else
        f=f(1:idstart-1);
        Yall=Yall(1:idstart-1,1:end);
        phi_in=phi_in(1:idstart-1,1:end);
    end
        phi_in=phi_in';
    
    
    %loop across models g(f)
    for idx=1:length(nlfvec)
        nlf=nlfvec(idx);
        Y=Yall(:,nlf);
        Ytest=Yalltest(:,nlf);
        
        % perform model optimisation and prediction
        [fhat,yhat,mse,nlpd,optimresults]=multitask_egp_ugp(nlf,Y,phi_in,Ytest,ftest,phi_intest)
        xvresults{c,idx}={fhat,yhat,mse,nlpd,optimresults};
        save('xvresults.mat','xvresults','-append');
        

    end
end