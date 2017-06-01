function [Gamma,Gammasum,Xi]=hsinference(Xtrain,T,N,hmm)
% [Gamma,Gammasum,Xi]=hsinference(Xtrain,T,N,hmm,options)
%
% inference engine for HMMs.
% 
% INPUT
%
% Xtrain    observation sequence
% T         length of observation sequence
% N         number of blocks
% hmm       hmm data structure
% options
%        .inftype    type of inference: 
%                    fwbw: foward backward
%                    gibsamp: gibbs sampling
%
% OUTPUT
%
% Gamma     Probability of hidden state given the data
% Gammasum  sum of Gamma over t
% Xi        joint Prob. of child and parent states given the data


inftype=hmm.train.inftype;

switch lower(inftype),
 
 case {'fwbw'},
  
  K=hmm.K;
  Gamma=[];
  Gammasum=zeros(1,K);
  Xi=zeros(T-1,K,K);
  lScale=zeros(T,1);
  
  for n=1:N
    [gamma,gammasum,xi,scale]=nodecluster(Xtrain,K,T,n,hmm);
    lScale=lScale+log(scale);
    Gamma=cat(1,Gamma,gamma);
    Gammasum=Gammasum+gammasum;
    Xi=Xi+reshape(xi,T-1,K,K);
  end;
 case{'gibsamp'},
  
  K=hmm.K;
  Gamma=[];
  Gammasum=zeros(1,K);
  Xi=zeros(T-1,K,K);
  lScale=zeros(T,1);
  
  for n=1:N
    [gamma,gammasum,xi,scale]=gibbssampling(Xtrain,K,T,n,hmm);
    lScale=lScale+log(scale);
    Gamma=cat(1,Gamma,gamma);
    Gammasum=Gammasum+gammasum;
    Xi=Xi+reshape(xi,T-1,K,K);
  end;
  
 otherwise
  error('Unknown or incorrect inference type')
end;

return;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Gamma,Gammasum,Xi,scale]=nodecluster(Xtrain,K,T,n,hmm)
% inference using normal foward backward propagation

P=hmm.P;
Pi=hmm.Pi;

B = obslike(Xtrain,T,n,hmm);

scale=zeros(T,1);
alpha=zeros(T,K);
beta=zeros(T,K);

alpha(1,:)=Pi.*B(1,:);
scale(1)=sum(alpha(1,:));
alpha(1,:)=alpha(1,:)/scale(1);
for i=2:T
  alpha(i,:)=(alpha(i-1,:)*P).*B(i,:);
  scale(i)=sum(alpha(i,:));		% P(X_i | X_1 ... X_{i-1})
  alpha(i,:)=alpha(i,:)/scale(i);
end;

beta(T,:)=ones(1,K)/scale(T);
for i=T-1:-1:1
  beta(i,:)=(beta(i+1,:).*B(i+1,:))*(P')/scale(i); 
end;

Gamma=(alpha.*beta); 
Gamma=rdiv(Gamma,rsum(Gamma));
Gammasum=sum(Gamma);

Xi=zeros(T-1,K*K);
for i=1:T-1
  t=P.*( alpha(i,:)' * (beta(i+1,:).*B(i+1,:)));
  Xi(i,:)=t(:)'/sum(t(:));
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Gamma,Gammasum,Xi,scale]=gibbssampling(Xtrain,K,T,n,hmm)
% inference using stochastic simulation

P=hmm.P;
Pi=hmm.Pi;
Nsamp=hmm.train.Nsamp;

B = obslike(Xtrain,T,n,hmm);

scale=zeros(T,1);


S=multinomrnd(Pi,T);			% intialisation of all hidden states
[tmp,S]=max(S,[],1);

Gamma=zeros(T,K);
Xi=zeros(T-1,K*K);

for n=1:Nsamp,
  t=1;
  Gamma(t,:)=B(t,:).*P(:,S(t))'.*Pi;
  scale(t)=sum(Gamma(t,:));
  Gamma(t,:)=Gamma(t,:)./scale(t);
  S(:,t)=find(multinomrnd(Gamma(t,:),1)==1);
  
  for t=2:T-1,
    Gamma(t,:)=B(t,:).*P(S(t-1),:).*P(:,S(t+1))';
    scale(t)=sum(Gamma(t,:));
    Gamma(t,:)=Gamma(t,:)./scale(t);
    Xi(t-1,:)=zeros(1,K*K);
    Xi(t-1,sub2ind([K,K],S(t-1),S1(t)))=1;
    S(t)=find(multinomrnd(Gamma(t,:),1)==1);
  end;
  t=T;
  Gamma(t,:)=B(t,:).*P(S(t-1),:);
  scale(t)=sum(Gamma(t,:));
  Gamma(t,:)=Gamma(t,:)./scale(t);
  S(:,t)=find(multinomrnd(Gamma(t,:),1)==1);
  Xi(t-1,:)=zeros(1,K*K);
  Xi(t-1,sub2ind([K,K],S(t-1),S1(t)))=1;
end;


Gammasum=sum(Gamma);