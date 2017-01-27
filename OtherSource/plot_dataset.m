close all;clc;clear;

load question1
f = figure;

hold on
plot(xtrain,ytrain)
plot(xtest,ones(87,1)*-35)
text(16.5,-33,'?','Color','red','fontsize',25)
box on

xlabel('xtrain')
ylabel('ytrain')
xlim([-33 22])

set(gca,'fontsize',20)

saveas(f, './figure/data1.fig')
saveas(f, './figure/data1.epsc')

