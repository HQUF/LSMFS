function [rank] = LSMFS(X,Y,alpha,beta,gamma,lambda)

    [nFeat,nSamp] = size(X);
    [~,nLabel]=size(Y);

    W = rand(nFeat,nLabel);
    V = rand(nSamp,nLabel);
    M= rand(nSamp,nLabel);
    B=Construct_B(Y);
    S = rand(nSamp,nLabel);
    I =  eye(nLabel);
    
    para.k=size(Y,2)-1;
    Ly = Laplacian_GK1(Y,para);
    D=eye(nFeat);
    
    
    
    
    Niter=21; %10
    err=1;
    iter=1;
    while (err > 10^-6 & iter< Niter)

    M=Y+B.*S;
    
    [U,~,P]=svd(W,'econ');
    %% W V 
    W = inv(X*X'+gamma*D)*(X*V-lambda*U*P');
    V = (X'*W+beta*M)*inv(alpha*I+beta*Ly'+beta*Ly);
    
    %% S
    S_original=B.*(V-Y);
    [x11,y11]=size(Y);
    for i=1:x11
        for j=1:y11
            S(i,j)=max(S_original(i,j),0);
        end
    end  
    
    wi = sqrt(sum(W.*W,2));
    d = 0.5./(wi+eps);
    D = diag(d);

    obj(iter)= norm(X'*W-V,'fro')^2+alpha*norm(V-M,'fro')^2+beta*trace(V*Ly*V')+gamma*trace(W'*D*W)+lambda*trace(sqrt((W*W')));
    if iter>1
            err = abs(obj(iter-1)-obj(iter));
    end
    iter=iter+1;
    end
    [~, rank] = sort(sum(W.*W, 2), 'descend');
    
end



function B=Construct_B(Y)
%%
[x1,y1]=size(Y);
B_origin=Y;
for i=1:x1
    for j=1:y1
      if Y(i,j)==0
        B_origin(i,j)=-1;
      end
    end
end
B=B_origin;
end