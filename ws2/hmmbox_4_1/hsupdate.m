function [hmm]=hsupdate(Xi,Gamma,T,N,hmm)

% updates hidden state parameters of an HMM
% 
% INPUT:
%
% Xi     probability of past and future state cond. on data
% Gamma  probability of current state cond. on data
% K      state space dimension
% T      length of observation sequence
% N      number of blocks
% hmm    single hmm data structure
%
% OUTPUT
% hmm    single hmm data structure with updated state model probs.

K=hmm.K;
% transition matrix 
sxi=squeeze(sum(Xi,1));   % counts over time

hmm.Dir2d_alpha=sxi+hmm.prior.Dir2d_alpha;
PsiSum=psi(sum(hmm.Dir2d_alpha(:),1));
for j=1:K,
  for i=1:K,
    P(j,i)=exp(psi(hmm.Dir2d_alpha(j,i))-PsiSum);
  end;
  P(j,:)=P(j,:)./sum(P(j,:));
end;
hmm.P=P;

% intial state
hmm.Dir_alpha=hmm.prior.Dir_alpha;
for i=1:N
  hmm.Dir_alpha=hmm.Dir_alpha+Gamma((i-1)*T+1,:);
end
PsiSum=psi(sum(hmm.Dir_alpha,2));
for i=1:K,
  Pi(i)=exp(psi(hmm.Dir_alpha(i))-PsiSum);
end
hmm.Pi=hmm.Pi./sum(hmm.Pi);
  
