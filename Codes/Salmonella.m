clear all
close all
clc
addpath(genpath('/Users/Mehraveh/Documents/'))

figure_number = 5

load(['/Users/Mehraveh/Documents/Reza/Salmonella/imp_feature_Figure', num2str(figure_number),'.mat'])
load(['/Users/Mehraveh/Documents/Reza/Salmonella/Acc_Figure', num2str(figure_number),'.mat'])
load(['/Users/Mehraveh/Documents/Reza/Salmonella/label.mat'])

Salmonella_data = csvread('/Users/Mehraveh/Documents/Reza/Salmonella/Species09042018_clean_PTMs.csv');

samples = {'S. cerevesiae','K. aerogenes', 'HEK', 'HeLa', 'S typhimurium',...
    'E coli', 'H Salinarium','Contaminated Milk', 'Uncontaminated Milk', ...
    'Arabidopsis', 'S. cerevesiae-trf4'}

%%%%%% Figure 1
if figure_number == 1
    classnumbers = [1,4,5,7,10]    
elseif figure_number == 2
    classnumbers = [5,6,2]
elseif figure_number == 3
    classnumbers = [3,4]
elseif figure_number == 4
    classnumbers = [1,11]            
elseif figure_number == 5
    classnumbers = [1:11]            
end

samples = samples(classnumbers);

% A
figure
accs = [precision;recall]'
cmap = othercolor('Spectral4',length(precision))
% cmap = lines(length(precision))
c = [[0,0,0];[0.5,0.5,0.5]]
x1 = -0.15:1:length(precision)
x2 = 0.15:1:length(precision)+0.25
for i = 1:size(accs,1)
    b = bar(x1(i),accs(i,1),0.25,'FaceColor',cmap(i,:),'EdgeColor','none','LineWidth',1.5),hold on 
    b = bar(x2(i),accs(i,2),0.25,'FaceColor',cmap(i,:),'EdgeColor','none','LineWidth',1.5),hold on 
    alpha(b,0.5)
end


clear h
h(1) = bar(NaN,NaN,'FaceColor',[0,0,0]);
h(2) = bar(NaN,NaN,'FaceColor',[0.5,0.5,0.5]);
legend(h,{'Precision';'Recall'});
% axis square
set(gca,'XTick',0:1:length(precision)-1)
set(gca,'XTickLabelRotation',45)
set(gca,'XTickLabel',samples)

set(gca,'box','off')
set(gca,'Ticklength', [0 0])
set(gca,'FontSize',30); 
set(gca,'FontWeight','normal');
color = get(gca,'Color');
grid on

ylim([0.5,1.05])
xlim([-0.75,length(classnumbers)])
ylabel('Identification Accuracy') 

% B
figure
[~,j]=ismember(Salmonella_data(:,1),classnumbers)
Salmonella_data(j==0,:)=[];
l = size(Salmonella_data,1)
V = Salmonella_data(1:l,2:end)';


[sorted_imp, idx] = sort(imp,'descend')
label_sorted = label(idx,:)
V_selected = V(idx(1:2),:)
for i = 1:size(V,2) 
    tmp=Salmonella_data(i,1);    
    s=scatter(V_selected(1,i),V_selected(2,i),500,'MarkerFaceColor',cmap(find(classnumbers==tmp),:),'MarkerEdgeColor','none'), hold on 
    alpha(s,0.8)
end

clear h
for i = 1:length(precision)
    h(i) = plot(NaN,NaN,'.','color',cmap(i,:),'MarkerSize',50), hold on  
end


legend(h,samples');
set(gca,'box','off')
axis('square')
set(gca,'Ticklength', [0 0])
set(gca,'FontSize',25); 
set(gca,'xticklabelrotation',45); 
set(gca,'FontWeight','normal');
color = get(gca,'Color');
grid on

xlabel(sprintf('First important PTM (%s)',label_sorted(1,find(~isspace(label_sorted(1,:))))));
if imp(idx(2)~=0)
    ylabel(sprintf('Second important PTM (%s)',label_sorted(2,find(~isspace(label_sorted(2,:))))));
else
    ylabel(sprintf('Randomly chosen PTM (%s)',label_sorted(2,find(~isspace(label_sorted(2,:))))));
end

xlim([min(V_selected(1,:))-range(V_selected(1,:))/10,max(V_selected(1,:))+range(V_selected(1,:))/10])
ylim([min(V_selected(2,:))-range(V_selected(2,:))/10, max(V_selected(2,:))+range(V_selected(2,:))/10])
% yruler = ax1.YRuler;
% yruler.Axle.Visible = 'off';
% xruler = ax1.XRuler;
% xruler.Axle.Visible = 'off';
set(gca,'color','none')

% [~, I] = unique(C); %
% I = length(C) - I(length(I):-1:1); 
% p = findobj(gca,'Type','Patch');
% legend(p(I),'X','Y','Z');


% C

for i=1:length(label)    
    label_cell{i} = label_sorted(i,find(~isspace(label_sorted(i,:))));
end
figure
cmap = (jet)
barh(sorted_imp)
for i = 1:size(sorted_imp,2)
    b = bar(i,sorted_imp(i),0.75,'FaceColor',cmap(i,:),'EdgeColor','none','LineWidth',1.5),hold on         
end


set(gca,'XTick',1:size(sorted_imp,2))
set(gca,'XTickLabelRotation',45)
set(gca,'XTickLabel',label_cell)
set(gca,'box','off')
set(gca,'Ticklength', [0 0])
set(gca,'fontsize',12) 
set(gca,'FontWeight','normal');
color = get(gca,'Color');
grid on
ylabel('Feature Importance','fontsize',25) 
xlabel('Features (PTMs)','fontsize',25)


