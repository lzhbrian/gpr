% cd /Users/Brian/Desktop/gpr
% addpath(genpath('/Users/Brian/Desktop/gpr/gpml-matlab-v4.0-2016-10-19'))
% addpath(genpath('/Users/Brian/Desktop/gpr/gauss_regression'))

clear; clc; close all;
tic

s = RandStream('mt19937ar','Seed',100);
RandStream.setGlobalStream(s);

%% Data Set
load planecontrol.mat
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


%% Calc MSE
MSE_plane_control(ytest_mu)


toc


