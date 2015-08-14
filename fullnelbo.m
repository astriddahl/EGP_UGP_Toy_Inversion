function [fullF]=fullnelbo(hparams,M,C,phi_in,Y,dims,nlf,LAMBDA,theta)
% Calculation of Negative Evidence Lower Bound (nelbo) - full expression
% full expression for nelbo = eqn(8)-eqn(5)
% A Dahl August 2015

%hparams=[diag(DELTA)', theta];

% g(Wphi)              - nonlinear transform (nlf) of f=Wphi and associated
%                        Jacobian J(f=M*phi_n)
    switch nlf
        case 1; g=@(f) f; 
                J=@(f) 1;    
        case 2; g=@(f) f+f.^2+f.^3; 
                J=@(f) 1+2*f+3*f.^2; 
        case 3; g=@(f) exp(f);
                J=@(f) exp(f);
        case 4; g=@(f) sin(f);
                J=@(f) cos(f);
        case 5; g=@(f) tanh(f);
                J=@(f) sech(f).^2;
    end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % function to calculate F
    
    % assign local values
    dims=num2cell(dims);
    [N,D,P,Q]=deal(dims{:});
    DELTA=diag(hparams(1:P));
    IDEL=1/DELTA;
    %theta=hparams(P+1:end);
    
    % generate PHI cos and sin features
    PHI(1:D/2,:)=cos(phi_in/theta);
    PHI(D/2+1:D,:)=sin(phi_in/theta);
    PHI=sqrt(2/D)*PHI;       %reestimate phi with new hparams. Note scaling must be (D/2)^-0.5 (features are duples)
        
    % calculation of F split into F=F1+F2+F3 corresponding to three terms
    % comprising eqn(8)-eqn(5)
    F1=0;
    
       for z=1:N                        % calculate Fn (sum elements of eqn 26)
        phi_n=PHI(:,z);
        f=M*phi_n;
        gn=g(f);                        %g maps Qx1 to Px1 - both scalar at present
        J_n=J(f);                       %"Jacobian" dg/df
        b_n=gn-J_n*f;
        y_n=Y(z,:)';
        vecn=y_n-J_n*f-b_n;
        Fn=vecn'*IDEL*vecn;
            % for Q=1 only:
            tr=trace(phi_n*J_n'*IDEL*J_n*phi_n'*C);
        Fn=Fn+tr;
        F1=F1+Fn;                       %cumsum over Fn=sum (over n) first term in eqn 26
       end

    F2=N*(P*log(2*pi)+log(prod(hparams(1:P)))); %prod(hparams)=det(DELTA)
    F3=0;
    for q=1:Q
        mq=M(q,:);
        Cq=C;               %implemented for Q=1 only: fix to pull out relevant diagonal block  from C
        Fq=1/(LAMBDA(q,q))*(trace(Cq)+(mq*mq'))-log(det(Cq))+D*log(LAMBDA(q,q))-D;
        F3=F3+Fq;
    end
    fullF=-0.5*sum([F1,F2,F3]);  % free energy
    
           
end
