clear;clc;close all;
s = RandStream('mt19937ar','Seed',0);
RandStream.setGlobalStream(s);

load question1.mat

% xtrain, ytrain
% xtest


%% Mean Function
meanfunc = []; hyp.mean = [];
meanfunc = @meanConst; hyp.mean = [0];
meanfunc = @meanLinear; hyp.mean = [0]; % 0.2112
% meanfunc = {@meanPoly,2}; hyp.mean = [0,0]; % 1.3
% meanfunc = {@meanPoly,3}; hyp.mean = [0,0,0]; % 0.2476
% meanfunc = {@meanPoly,4}; hyp.mean = [0,0,0,0]; % wtf
% meanfunc = {@meanPoly,5}; hyp.mean = [0,0,0,0,0]; % wtf


%% Covariance Function
ell = .9;
sf = 2;
p = 2;
al = 2;

covfunc = {@covSum,{{@covProd,{@covLINiso,@covSEiso}},...
          {@covProd,{@covSEiso,{@covSum,{@covPeriodic,@covRQiso}}}}}};
%              LIN,           SE,           SE,           PER,            RQ,  
% hyp.cov = [[log(rand(1))];log([ell;sf]);log([ell;sf]);log([ell;p;sf]);log([ell;sf;al])];
hyp.cov = [[0];           [0;0];        [0;0];        [0;0;0];        [0;0;0]];


% covfunc = {@covProd,{@covPeriodic,{@covSum,{@covPeriodic,@covLINiso}}}};
% %              PER,            PER,            LIN
% hyp.cov = [log([ell;p;sf]);log([ell;p;sf]);[log(rand(1))]];
% hyp.cov = [[0;0;0];        [0;0;0];        [0]];


%% Likelihood Function
sn = 0.1;
likfunc = @likGauss;              % Gaussian likelihood
hyp.like = [log(sn)];
hyp.like = [0];


%% Inference Method
inffunc = @infGaussLik;


%% Minimize
hyp = struct('mean', hyp.mean, 'cov', hyp.cov, 'lik', hyp.like);
hyp2 = minimize(hyp, @gp, -500, inffunc, meanfunc, covfunc, likfunc, xtrain, ytrain);


%% Predict
[ytest_mu ytest_s2] = gp(hyp2, inffunc, meanfunc, covfunc, likfunc, xtrain, ytrain, xtest);


%% Plot
f = [ytest_mu+2*sqrt(ytest_s2); flipdim(ytest_mu-2*sqrt(ytest_s2),1)];
fill([xtest; flipdim(xtest,1)], f, [7 7 7]/8)
hold on; plot(xtest, ytest_mu); 
plot(xtrain, ytrain,'b')
xlim([-33 22])


%% Calc MSE
MSE_question2(ytest_mu)

