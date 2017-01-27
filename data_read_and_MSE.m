clear

addpath(genpath('gpml/'))

%%%%%%%%%%%%%%%%
%% ????1
clear;
load('question1.mat');
% ??????????????1??????

%% Dataset
s = RandStream('mt19937ar','Seed',0);
RandStream.setGlobalStream(s);

% load question1.mat
% xtrain, ytrain
% xtest


%% Mean Function
% meanfunc = []; hyp.mean = [];
% meanfunc = @meanConst; hyp.mean = [0];
meanfunc = @meanLinear; hyp.mean = [0]; % 0.2112
% meanfunc = {@meanPoly,2}; hyp.mean = [0,0]; % 1.3
% meanfunc = {@meanPoly,3}; hyp.mean = [0,0,0]; % 0.2476
% meanfunc = {@meanPoly,4}; hyp.mean = [0,0,0,0]; % wtf
% meanfunc = {@meanPoly,5}; hyp.mean = [0,0,0,0,0]; % wtf


%% Covariance Function
% some init num
ell = .9;
sf = 2;
p = 2;
al = 2;
covfunc = {@covSum,{{@covProd,{@covLINiso,@covSEiso}},...
          {@covProd,{@covSEiso,{@covSum,{@covPeriodic,@covRQiso}}}}}};
%              LIN,           SE,           SE,           PER,            RQ,  
% hyp.cov = [[log(rand(1))];log([ell;sf]);log([ell;sf]);log([ell;p;sf]);log([ell;sf;al])];
hyp.cov = [[0];           [0;0];        [0;0];        [0;0;0];        [0;0;0]];

% Another possible Ker, unused
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


% Calc MSE
mse = MSE_question2(ytest_mu)





%%%%%%%%%%%%%%%%
%% ????2
clear;
load('planecontrol.mat');
% ??????????????2??????
tic

s = RandStream('mt19937ar','Seed',100);
RandStream.setGlobalStream(s);


%% Data Set
% load planecontrol.mat
% xtrain, ytrain
% xtest

% ignore some highly correlated dimensions (>0.96)
    % 11 == [12:24,39,40]
    used_dimension = [1:11,25:38];
xtrain = xtrain(:,used_dimension);
xtest = xtest(:,used_dimension);

% sample num of data
num = 1000;
sampled_no = randsample(1:10000, num);
xtrain = xtrain( sampled_no ,:);
ytrain = ytrain( sampled_no ,:);


%% Mean Func
% meanfunc = []; hyp.mean = [];
meanfunc = @meanConst;      hyp.mean = [0];


%% Cov Func
D = length(used_dimension);
R = D; % 5;
% some init values
ell = .9;
sf = 2;
p = 2;
al = 2;
L = rand(D,1);
covfunc = { 'covSEard' };
% hyp.cov = [ log([L;sf]) ];
hyp.cov = [ zeros(D,1);0 ];


%% Like Func
likfunc = @likGauss;        hyp.like = [log(0.1)];
% likfunc = @likLaplace;      hyp.like = [log(0.1)];


%% Inf Func
% inffunc = @infGaussLik;
inffunc = @infLaplace;
% inffunc = @infVB; 
% inffunc = @infLOO;
% Not using
    % % inffunc = @infPrior;
    % % inffunc = @infEP;
    % % inffunc = @infKL;
    % % inffunc = @infMCMC;


%% Minimize
hyp_struct = struct('mean', hyp.mean, 'cov', hyp.cov, 'lik', hyp.like);
hyp2 = minimize(hyp_struct, @gp, -500, inffunc, meanfunc, covfunc, likfunc, xtrain, ytrain);


%% Predict
[ytest_mu, ytest_s2] = gp(hyp2, inffunc, meanfunc, covfunc, likfunc, xtrain, ytrain, xtest);


% Calc MSE
mse = MSE_plane_control(ytest_mu)
