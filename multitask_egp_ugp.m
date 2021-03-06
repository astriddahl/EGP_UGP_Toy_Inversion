function [fhat,yhat,fsmse,fnlpd,ysmse,optimresults]=multitask_egp_ugp(nlf,Y,phi_in,Ytest,ftest,phi_intest)
% Implementation of Multi-task Extended and Unscented Gaussian Process - a
% la carte (Steinberg and Bonilla)
% code by A Dahl 2015

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Notes
% data description etc
% NB notation follows paper above (multitask paper) except: LAMBDA matrix
%   renamed to DELTA to avoid confusion with matrix of diag(lambda_q)
% need to implement options for phi - currently set up for RBF
% need to implement Cq for multi Q - at present only for Q=1

% implement stat linearisation
% implement multi-P, multi-Q for nonlinear function g and gn (currently
% scalar only), M (currently M==mq where noted, needs updating to loop over
% q)
% implement calculation of J_n for multi-d (currently scalar input only)
% need to check whether D scaling for random features PHI needs
% check Jacobians for P>1
% fix gradient definitions for theta - capture indirect derivs
% check loop exits

% change names of nelbo and fullnelbo to freeE (since min free energy=F)
% 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Setup
% addpath(genpath('C:\Program Files\MATLAB\R2014b\astridlib')); %requires minfunc, opt toolbox and (for toyinvdata generation) gpmltoolbox
% addpath(genpath('C:\Users\adahl\Documents\MATLAB\EGP_UGP'));    %includes nelbo.m and fullnelbo.m
% include logging
%clear;
% definitions
% X Y raw input data - requires matrices Y(NxP) and X(Nxd)
    % *** at present no generic code for this - inputs from toyinvdata
    % section below *****
    

% g(Wphi)              - choose nonlinear transform (nlf) of f=Wphi (TBC)
%nlf=1;                %- now passed in from multitask_egp_prediction.m
    switch nlf
        case 1; g=@(f) f; 
                J=@(f) 1;    %Jacobian J(f=M*phi_n)
        case 2; g=@(f) f+f.^2+f.^3; 
                J=@(f) 1+2*f+3*f.^2; 
        case 3; g=@(f) exp(f);
                J=@(f) exp(f);
        case 4; g=@(f) sin(f);
                J=@(f) cos(f);
        case 5; g=@(f) tanh(2*f);
                J=@(f) 2*(sech(2*f).^2);
    end

% define function evaluating objective function F (eqn 26) at
% hyperparameters
% NB  THIS SHOULD HAVE WORKED! ADDITIONAL INPUTS NOW PASSED
% DIRECTLY THROUGH MINFUNC/VARARGIN
%fobj=@(hparams) nelbo(hparams,M,C,phi_in,Y,dims,nlf,LAMBDA);
%%



% define dimensions N (nobs), D (no features), P (no output), Q (no tasks)
% add line specifying number of features to generate (D) and N=length x
N=size(phi_in,2);            % NxD feature input matrix
D=2*size(phi_in,1);          % doubled as PHI creates both sin and cos from each D_input
P=size(Y,2);                 % is 1 for toyinvdata Y(Nx1)
Q=1;                        % choose number of tasks Q for multitask model
dims=[N,D,P,Q];
% phi               - choose feature map (for now set to random gaussian features from toyinvdata)
% method            - choose MAP/grad descent or stat linearisation
                      % ***at present implemented for grad desc only
% predict           - choose whether to perform prediction step (1=yes/0=no)
predict=1; 
maxiter=50;           % - choose max iterations (currently assigned for both overall loop and M/C loop)
conv=0.001;                %- choose convergence threshold (for params)
deltam=1;           %initialise convergence measure for inner loop > conv
% setseed           - choose rng seed
%nsamples=10000;     % as per paper 1 (currently not used)
a=0.9;                % choose initial learning rate for grad descent


% starting values - choose M0, delta0, lamda0, theta0
M0=2;             % initialised for all mq
M=M0*ones(Q,D);     % approx mean vec for each task Q, feature D. mq=M(q,:) - 1xD
d0=2;
DELTA=d0*diag(ones(P,1));  % note this is actually the matrix LAMBDA in multitask egp paper
lam0=1;             % LAMBDA (set to diag vii=1)
LAMBDA=lam0*diag(ones(Q,1));         %lamda(q,q)=lamdaq
th0=0.9;
theta=th0;           % single parameter sigma for random gaussian features


% set options for minfunc solver
options=struct('numDiff',0,'DerivativeCheck','on','Method','lbfgs');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PARAMETER ESTIMATION
% parameter estimates for approximate posterior m,C,
% and hyperparameters 

%% Initialise C
%F outer loop requires initial values for PHI,J,M,DELTA,LAMBDA. M0,lam0,del0 defined
%above. This section adds PHI (at theta0) and J (at M0,PHI) and evaluates C at M0.

PHI=zeros(D,N);     % generate initial PHI cos and sin features (at theta0)
    PHI(1:D/2,:)=cos(phi_in/theta);
    PHI(D/2+1:D,:)=sin(phi_in/theta);
    PHI=sqrt(2/D)*PHI;
%J_n=zeros(P,Q);      
% initialise C matrix at M0
H=zeros(D,D);
for z=1:N                                       % calculate Fn (sum elements of eqn 26)
        phi_n=PHI(:,z);
        f=M*phi_n;
        J_n=J(f);                               %"Jacobian" dg/df
        Hinner=J_n'*(1/DELTA)*J_n;              % Hn broken into two steps to improve stability of calculation
        Hn=phi_n*Hinner*phi_n';    % implemented for Q=1, P=1 only (eqn(11))-should extract q column of J_n
        H=H+Hn;                                 %add Hn to cumulative sum over n to be output from n-loop - implemented for Q=1 only     
end
%%
ILAM=1/LAMBDA;   % lambda_q are treated as fixed priors and not optimised in this model
H=-H-ILAM*eye(D);            % needs review for Q>1 - won't work
C=-inv(H);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Step 1. Loop to optimise hyperparameters given m - set up for numerical optimisation

% track objective function value
Fouter=zeros(maxiter^2,1);
Finner=zeros(maxiter^2,1);
% initial values hyperparameters
% hparams0=[diag(DELTA)', theta]';  fixing theta at true value
hparams0=diag(DELTA);
for w=1:maxiter 
    if w==1
        hparams=hparams0;
        deltah=1;           % initialise convergence threshold for hparams
    end
    
   
%         [x,f,exitflag,output]=minFunc(@nelbo,hparams,options,M,C,phi_in,Y,dims,nlf,LAMBDA,theta);
%         if isreal(f)==0
%             display('F complex');
%             return;
%         end
%         deltah=norm(abs(x-hparams))
%         Fouter(w)=f;
%         plot(Fouter(w),'-d');        

    
    
    if  deltah>conv        %deltah not defined for fixed hparams
        display(w);
        %redefine hyperparameters based on hparamest
        display(hparams);
        DELTA=diag(hparams(1:P));
        IDEL=1/DELTA;
            % For free theta:
            %theta=hparams(P+1:end);
    %         update PHI
                PHI=zeros(D,N);
                PHI(1:D/2,:)=cos(phi_in/theta);
                PHI(D/2+1:D,:)=sin(phi_in/theta);
                PHI=sqrt(2/D)*PHI;

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %Step 2. Loop to optimise M: hyperparameters fixed at DELTA/IDEL, LAMBDA/ILAM, theta
            % for Q>1 insert loop over q here
            mk1=M';                      %for Q=1 only NB M is QxD but each mq is Dx1       
            dd=-1;                       % initialise divergence measure.
            deltam=1;                    % reset convergence measure for m
            for v=1:maxiter             %loop to optimise M: hyperparameters fixed at DELTA/IDEL, LAMBDA/ILAM, theta
                display(v);
            if deltam>conv %&& dd<1           %dd divergence threshold should be zero really
                mk=mk1;
                dF=zeros(D,1);          % may need fixing for Q>1
                H=zeros(D,D);
                for z = 1:N             % update 1:N
                    y_n=Y(z,:)';
                    phi_n=PHI(:,z);     %phi(xn) Dx1 vec
                    f=phi_n'*mk;
                    gn=g(f);                        %g maps Qx1 to Px1 - both scalar at present
                    J_n=J(f);                       %"Jacobian" dg/df
                    dFn=J_n*IDEL*(y_n-gn)*phi_n; % for multitask probably needs amending from eqn (15): phi_n to phi_n'
                    dF=dF+dFn;                     % add dFn to cumulative sum over n to be output from n-loop
                    Hinner=J_n'*IDEL*J_n;              % Hn broken into two steps to improve stability of calculation
                    Hn=phi_n*Hinner*phi_n';    % implemented for Q=1, P=1 only (eqn(11))-should extract q column of J_n
                    H=H+Hn;                      %add Hn to cumulative sum over n to be output from n-loop - implemented for Q=1 only
                end
                dF=dF-ILAM*mk;
                H=-H-ILAM*eye(D);            % needs review for Q>1 - won't work
                
                %learning step
                mk1=mk-a^(v/2)*inv(H)*dF;       %learning rate decays slowly


                % report F at each iteration
                M=mk1';
                    H=zeros(D,D);
                    for z=1:N                                       
                    phi_n=PHI(:,z);
                    f=M*phi_n;
                    J_n=J(f);                               %"Jacobian" dg/df
                    Hinner=J_n'*IDEL*J_n;              % Hn broken into two steps to improve stability of calculation
                    Hn=phi_n*Hinner*phi_n';    % implemented for Q=1, P=1 only (eqn(11))-should extract q column of J_n
                    H=H+Hn;                                 %add Hn to cumulative sum over n to be output from n-loop - implemented for Q=1 only     
                    end
                    H=-H-ILAM*eye(D);            % needs review for Q>1 - won't work
                    C=-inv(H);                  % NB implemented for Q=1 only
                
                fullF=fullelbo(hparams,M,C,phi_in,Y,dims,nlf,LAMBDA,theta)
                    if isreal(fullF)==0
                    display('F complex');
                    return;
                    end
                dd1=deltam;                  % carry over prev value
                deltam=norm(abs(mk1-mk))    %update deltam
                dd=deltam-dd1                %calc change in deltam (ie check not diverging)
                display('average M');
                display(mean(M));
                Finner(v*w)=fullF;
                

            end % m+converged. If deltam below conv on last loop, M+ stops at mk1
                if deltam<conv
                    break;                  %do not complete remaining w iters
                end
            end
            %end of loop to optimise M
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % re-optimise hyperparameters using updated optimised M+ and C+ 


        display('Updating NELBO with converged M+, C+...');
        [x,f,exitflag,output]=minFunc(@nelbo,hparams,options,M,C,phi_in,Y,dims,nlf,LAMBDA,theta);
        if isreal(f)==0
            display('F complex');
            return;
        end
            
        deltah=norm(abs(x-hparams))       
        hparams=x               % parameter update
        Fouter(v*w)=f;
        

    end
    
    if deltam<conv  && deltah<conv 
        display('converged');            
        optresult='converged';
        break;                  %do not complete remaining w iters
    end
end

%end of loop to optimise hyperparameters and M,C
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%

% split data into training and test data? or set up j-fold validation

% MAP objective and simple gradient descent

% statistical linearisation



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Prediction
%implemented for P=1, Q=1 only
if predict==1
   %generate [fhat,yhat,mse,nlpd] using Ytest,ftest,phi_intest
   Y=Ytest;
   phi_in=phi_intest;
   f=ftest;
   sdY=std(Y);
   sdf=std(f);
        %update PHI with phi_intest
                ntest=length(Y);
                PHI=zeros(D,ntest);
                PHI(1:D/2,:)=cos(phi_in/theta);
                PHI(D/2+1:D,:)=sin(phi_in/theta);
                PHI=sqrt(2/D)*PHI;
    % calculate fhat,yhat,P(fhat)
    fhat=zeros(ntest,1);
    yhat=zeros(ntest,1);
    fpd=zeros(ntest,1);
                for z=1:ntest                                       
                    phi_n=PHI(:,z);
                    fhatn=M*phi_n;
                    fhat(z)=fhatn;
                    yhat(z)=g(fhatn);
                    %J_n=J(fhatn);
                    %vecn=J_n*M*phi_in;
                    %v_yn=DELTA+J_n*phi_n'*C*phi_n*J_n';  %implemented for Q=1 only    
                    %p_fn=P(f_qn|phi_n,M,C)~N(phi_n'm,phi_n'Cphi_n)
                    v_fn=phi_n'*C*phi_n;
                    fpd(z)=normpdf(f(z),fhatn,v_fn);
                end
   % calculate MSME and NLPD scores
    ydiff=(Y-yhat)/sdY;             %normalised by Ytest sample std dev.     
    ysmse=mean(ydiff.^2);
    fdiff=(f-fhat)/sdf;
    fsmse=mean(fdiff.^2);
    fnlpd=-mean(log(fpd));
                      
end
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Evaluation/results/output
optimresults={M,C,hparams,Finner,Fouter,optresult};
%% end of function
end
