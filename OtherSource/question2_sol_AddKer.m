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
num = 200;
sampled_no = randsample(1:10000, num);
xtrain = xtrain( sampled_no ,:);
ytrain = ytrain( sampled_no ,:);


%% Mean Func
% meanfunc = []; hyp.mean = [];
meanfunc = @meanConst;      hyp.mean = [0];


%% Cov Func
ell = .9;
sf = 2;
p = 2;
al = 2;

D = length(used_dimension);
R = 3; % 5;
covfunc = { 'covADD',{1:R,'covSEiso'} };
hyp.cov = [ log(zeros(2*D,1)); log(zeros(R,1))];
% hyp.cov = [ ones(2*D,1); ones(R,1)];
hyp.cov = [ zeros(2*D,1); zeros(R,1)];


%% Like Func
likfunc = @likGauss;        hyp.like = [log(0.1)];
likfunc = @likLaplace;      hyp.like = [0];

%% Inf Func
                        % data:9901:10000
                        % R = 5; D = 40;
                        % covfunc = { 'covADD',{1:R,'covSEiso'} };
                        % hyp.cov = [ ones(2*D,1); ones(R,1)];
                        % meanfunc = @meanConst; hyp.mean = [0];
                        % likfunc = @likGauss; hyp.like = [0];
                        % minimize: -100
                                
                            % @likGauss  % @likLaplace
                        
inffunc = @infGaussLik;     % 0.0352     % N/A
inffunc = @infLaplace;      % 0.0351     % 0.0334
inffunc = @infVB;           % 0.0349
% inffunc = @infLOO;          % 0.0343     % 0.0828

% not using
    % inffunc = @infPrior;
    % inffunc = @infEP;
    % inffunc = @infKL;
    % inffunc = @infMCMC;

% reduced_dim, infLaplace, likGauss, R=5, num=500, -500 : 0.0302


%% Minimize
hyp_struct = struct('mean', hyp.mean, 'cov', hyp.cov, 'lik', hyp.like);
hyp2 = minimize(hyp_struct, @gp, -100, inffunc, meanfunc, covfunc, likfunc, xtrain, ytrain);


%% Predict
[ytest_mu, ytest_s2] = gp(hyp2, inffunc, meanfunc, covfunc, likfunc, xtrain, ytrain, xtest);


%% Calc MSE
MSE_plane_control(ytest_mu)

% Contribution
len = length(hyp.cov);
hyp2.cov((len-4):len);

toc


