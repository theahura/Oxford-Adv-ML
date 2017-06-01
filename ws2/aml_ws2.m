addpath('./netlab3_3/');
addpath('./hmmbox_4_1/');

clear;
x(1:500) = randn(1, 500);
x(501:1000) = randn(1, 500) + 10;
x(1001:1500) = randn(1, 500);
x(1501:2000) = randn(1, 500) + 20;
x(2001:2500) = randn(1, 500) + 10;
x(2501:3000) = randn(1, 500) - 10;
x(3001:3500) = randn(1, 500) + 10;
x(3501:4000) = randn(1, 500) - 10;
x(4001:4500) = randn(1, 500);
x = x';
% plot(x)

hmm.K=100;

T = size(x, 1);

disp(' ');
hmm=hmminit(x,hmm,'full');

disp('Means of HMM initialisation');
hmm.state(1).Mu
hmm.state(2).Mu

% Train up HMM on observation sequence data using Baum-Welch
% This uses the forward-backward method as a sub-routine
disp('We will now train the HMM using Baum/Welch');
disp(' ');

disp('Estimated HMM');

hmm.train.cyc=30;
hmm.obsmodel='Gauss';
hmm.train.obsupdate=ones(1,hmm.K);   
hmm.train.init=1;     

hmm=hmmtrain(x,T,hmm);
disp('Means');
hmm.state(1).Mu
hmm.state(2).Mu
disp('Initial State Probabilities, Pi');
hmm.Pi
disp('State Transition Matrix, P');
hmm.P

[block]=hmmdecode(x,T,hmm);
block().q_star

% Find most likely hidden state sequence using Viterbi method
figure
plot(block().q_star);
axis([0 T 0 100]);
title('Viterbi decoding');

disp('The Viterbi decoding plot shows that the time series');
disp('has been correctly partitioned.');