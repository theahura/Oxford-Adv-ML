function [hmm] = hmminit (X,hmm,covtype,gamma)

% function [hmm] = hmminit (X,hmm,covtype,gamma)
%
% Initialise  HMM Chain with Gaussian observation models
%
% X		N x p data matrix
% hmm		hmm data structure
% covtype       'full' or 'diag' covariance matrices
% gamma         weighting of each of N data points 
%               (default is 1 for each data point)
%
% e.g.   hmm=struct(K,2,'full');
%        hmm=hmmint(X,hmm); 

[T,ndim]=size(X);
if length(X)~=T,
  X=X';
  [T,ndim]=size(X);
end;

if nargin < 4 | isempty(gamma), gamma=ones(T,1); end
if nargin < 3 | isempty(covtype), covtype='full'; end


hmm=hmmhsinit(X,hmm);

hmm.obsmodel='Gauss';
obsoptions=struct('prrasc',1,'gamma',gamma,'covtype',covtype);

hmm=obsinit(X,hmm,obsoptions);

% for backward compatibility
for i=1:hmm.K,
  hmm.state(i).Mu=hmm.state(i).Norm_Mu;	
  hmm.state(i).Cov=hmm.state(i).Wish_B/(hmm.state(i).Wish_alpha-0.5*(ndim+1));
end