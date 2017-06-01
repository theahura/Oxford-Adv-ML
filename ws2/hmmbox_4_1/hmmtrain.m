function [hmm,FrEn]=hmmtrain(X,T,hmm)
% function [hmm,FrEn]=hmmtrain(X,T,hmm)
%
% Train Hidden Markov Model using using Variational Framework
%
% INPUTS:
%
% X - N x p data matrix
% T - length of each sequence (N must evenly divide by T, default T=N)
% hmm.K - number of states 
% hmm.P - state transition matrix
% hmm.obsmodel -  observation model
% hmm.train.cyc - maximum number of cycles of Baum-Welch (default 100)
% hmm.train.tol - termination tol (change in log-evidence) (default 0.0001)
% hmm.train.init - Already initialised the obsmodel (1 or 0) ? (default=0)
% hmm.train.obsupdate - Update the obsmodel (1 or 0) ?  (default=1)
% hmm.train.pupdate - Update transition matrix (1 or 0) ? (default=1)
%
% OUTPUTS
% hmm.Pi - initial state probability
% hmm.P - state transition matrix
% hmm.state(k).$$ - whatever parameters there are in the observation model
%
%
% OUTPUTS
% hmm.Pi          - intial state probability
% hmm.P           - state transition matrix
% hmm.state(k).$$ - whatever parameters there are in the
%                              observation model 
% hmm.FrEn        - free energy terms for each iteration
% hmm.Xi          - joint probability of past and future states
%                    conditioned on data 
% hmm.Gamma       -  probability of states conditioned on data 
% hmm.Gammasum    -  expectation of Gamma over time
%


% Copy in and check existence of parameters from hmm data structure

if ~isfield(hmm,'obsmodel')
  error('Error in hmm_train: obsmodel not specified');
  return
end

[N,ndim]=size(X);
if length(X)~=N,
  X=X';
  [N,ndim]=size(X);
end;

[hmm,K,cyc,tol,updateobs,updatep,rdisplay,runplot,plotoptions]= ...
    paramchk(X,T,hmm);



% Initialise stuff

if (rem(N,T)~=0)
  error('Error: Data matrix length must be multiple of sequence length T');
  return;
end;
N=N/T;

Pi=rand(1,K);
Pi=Pi/sum(Pi);

if ~isfield(hmm,'P')
  P=rand(K);
  P=rdiv(P,rsum(P));
else
  P=hmm.P;
end

FrEntrain=[];
FrEn=0;

alpha=zeros(T,K);
beta=zeros(T,K);
Gamma=zeros(T,K);
FrEntrain=[];

for cycle=1:cyc

  %%%% E step
  [Gamma,Gammasum,Xi]=hsinference(X,T,N,hmm);
  
  if runplot
    contplot(hmm,X,K,Gamma,plotoptions);
  end;

  %%%% M STEP 
  
  % transition matrices and initial state
  if updatep
    hmm=hsupdate(Xi,Gamma,T,N,hmm);
  end;

  % Observation model
  if sum(updateobs) > 0
    hmm=obsupdate(X,T,Gamma,Gammasum,hmm,updateobs);
  end

  %  evaluate free energy
  oldFrEn=FrEn;
  frEn=evalfreeenergy(X,Gamma,Xi,hmm);	% compute free energy
  FrEn=sum(frEn);
  FrEntrain=[FrEntrain; frEn];
  
  mesgstr='';
  if (cycle<=2)
    FrEnbase=FrEn;
  else
    if (FrEn-oldFrEn) > 0,
      mesgstr='(Violation)';
    end;
    if abs((FrEn - oldFrEn)/oldFrEn*100) < tol
      break;
    end;
  end;
  if rdisplay, 
    fprintf('cycle %i free energy = %f %s \n',cycle,FrEn,mesgstr);  
  end;

end

% for backward compatibility
for i=1:hmm.K,
  hmm.state(i).Mu=hmm.state(i).Norm_Mu;	
  hmm.state(i).Cov=hmm.state(i).Wish_B/(hmm.state(i).Wish_alpha-0.5*(ndim+1));
end

% for debugging etc
hmm.train.FrEn=FrEntrain;
hmm.train.Gamma=Gamma;
hmm.train.Gammasum=Gammasum;
hmm.train.Xi=Xi;
hmm.data.Xtrain=X;
hmm.data.T=T;

disp(sprintf('Model: %d kernels, %d dimension(s), %d data samples',...
	     K,ndim,T));
disp(sprintf('Final Free-Energy (after %d iterations)  = %f',...
	     cycle,FrEn)); 

return;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function contplot(hmm,data,K,Gamma,plotoptions)
%
% continous plotting of results
%

[nXgrid,nYgrid,Xgrid,Ygrid,Ncols,colstr,dpf,cpf]=deal(plotoptions{:});
[N,ndim]=size(data);

clf
if (dpf | cpf),
  [y,classndx]=max(Gamma,[],2);
  for k=1:K,
    if ndim>1
      plot(data(find(classndx==k),1),data(find(classndx==k),2), ...
	   colstr{rem(k,Ncols)+1}),hold on;
      centre=[hmm.state(k).Norm_Mu];
      Cov=hmm.state(k).Wish_B/hmm.state(k).Wish_alpha;
      text(centre(1),centre(2),sprintf('X-%s%d',blanks(k),k));
      if cpf
	for xg=1:nXgrid, 
	  for yg=1:nYgrid,
	    pdf(xg,yg)=gaussmd([Xgrid(xg,yg) Ygrid(xg,yg)],centre,Cov);
	  end;
	end;
	pdf=pdf./(max(max(pdf))-min(min(pdf)));
	contour(Xgrid(1,:),Ygrid(:,1),pdf,[.67 .67],'b:');
      end;
    else
      plot(find(classndx==k),data(find(classndx==k),1),...
	   colstr{rem(k,Ncols)+1}),hold on;
      centre=[hmm.state(k).Norm_Mu];
      Cov=hmm.state(k).Wish_B/hmm.state(k).Wish_alpha;
      plot(1:N,ones(1,N)*centre(1),colstr{rem(k,Ncols)+1});
      text(k,centre(1),sprintf('X-%s%d',blanks(k),k));
      if cpf
	for xg=1:nXgrid, 
	  for yg=1:nYgrid,
	    pdf(xg,yg)=gaussmd([Ygrid(xg,yg)],centre,Cov);
	  end;
	end;
	pdf=pdf./(max(max(pdf))-min(min(pdf)));
	contour(Xgrid(1,:),Ygrid(:,1),pdf,[.67 .67],'b:');
      end;
    end
  end;
  drawnow, hold off;
else
  T=length(Gamma);
  plot(Gamma);
  axis([0 T 0 1.1]);
  drawnow;
end;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [hmm,K,cyc,tol,updateobs,updatep,rdisplay,runplot,plotoptions]=paramchk(Xtrain,T,hmm);

%
% Copy in and check existence of parameters for hmm data structure
%
% Input hmm, data series X
%

if isfield(hmm,'K')
  K=hmm.K;
else
  error('Error in hmmtrain: K not specified');
  return
end


if ~isfield(hmm,'train')
  error('Error in hmmtrain: hmm.train not specified');
  return
end

if isfield(hmm.train,'cyc')
  cyc=hmm.train.cyc;
else
  cyc=100; 
end

if ~isfield(hmm.train,'inftype')
  hmm.train.inftype='fwbw'; 
end

if ~isfield(hmm.train,'Nsamp')
  hmm.train.Nsamp=K*30; 
end

if isfield(hmm.train,'tol')
  tol=hmm.train.tol;
else
  tol=0.001; 
end

if ~isfield(hmm.train,'obsupdate')
  updateobs=ones(1,hmm.K);  % update observation models for all states
else
  updateobs=hmm.train.obsupdate;
end

if ~isfield(hmm.train,'pupdate')
  updatep=1;
else
  updatep=hmm.train.pupdate;
end

if ~isfield(hmm.train,'rdisplay')
  rdisplay=1;
else
  rdisplay=hmm.train.rdisplay;
end

if ~isfield(hmm.train,'plot')
  runplot=0;
else
  runplot=hmm.train.plot;
end

if runplot
  % grid for plotting contours
  dmin=min(X);
  dmax=max(X);
  dspace=range(X)./30;
  if length(dmin)>1,
    [Xgrid,Ygrid] = meshgrid(dmin(1):dspace(1):dmax(1),dmin(2): ...
			     dspace(2):dmax(2));
    [nXgrid,nYgrid]=size(Xgrid);
  else
    [Xgrid,Ygrid] = meshgrid(1:N/30:N,dmin(1):dspace(1):dmax(1));
    [nXgrid,nYgrid]=size(Xgrid);
  end
  colstr={'y.';'m.';'c.';'r.';'g.';'b.';'k.'};
  Ncols=length(colstr);
  dpf=(runplot==2);
  cpf=(runplot==3);
  plotoptions{1}=nXgrid;
  plotoptions{2}=nYgrid;
  plotoptions{3}=Xgrid;
  plotoptions{4}=Ygrid;
  plotoptions{5}=Ncols;
  plotoptions{6}=colstr;
  plotoptions{7}=dpf;
  plotoptions{8}=cpf;
  figure;
else
  plotoptions={};
end;