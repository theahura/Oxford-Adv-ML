% A demonstration of the HMM software using a Gaussian observation
% model on AR features
clear all

load demdirichlet
load chmmsim
Xtrain=data.Xseries;
T=length(Xtrain);

hmm=struct('K',2);

disp(' ');
hmm=hmmhsinit(Xtrain,hmm);
hmm.obsmodel='Dirichlet';
obsoptions=struct('prrasc',1);
hmm=obsinit(Xtrain,hmm,obsoptions);

hmm.train.cyc=30;
hmm.train.obsupdate=ones(1,hmm.K);    % update observation models ?
hmm.train.init=1;         % Yes, we've already done initialisation

hmm=hmmtrain(Xtrain,T,hmm);

[block]=hmmdecode(Xtrain,T,hmm);

% Find most likely hidden state sequence using Viterbi method
figure
plot(block(1).q_star);
axis([0 T 0 3]);
title('Viterbi decoding');

