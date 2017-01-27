clear; clc; close all;
load planecontrol.mat

% xtrain, ytrain
% xtest

disp('Start\n')

dat = corr(xtrain);

count = 0
for i = 1:40
    for j = (i+1):40
        if abs(dat(i,j)) >= 0.96
            sprintf('(%d,%d)',i,j)
            count = count + 1;
        end
    end
end
disp(count)

figure;
bar3(dat)
zlabel('Pearson Correlation')
xlabel('Dimensions')
ylabel('Dimensions')
xlim([0 41])
ylim([0 41])

set(gca,'fontsize',20)





