clc
clear all
close all
addpath(genpath('/Users/Mehraveh/Documents/MATLAB/'))
load /Users/Mehraveh/Documents/MATLAB/Reza/probability_all_9.mat
figure

mat = prob_loo;  
imagesc(mat);
CC = (othercolor('Spectral4'))
colormap(CC)

% colorbar
% textStrings = num2str(mat(:),'%0.1f');  %# Create strings from the matrix values
% textStrings = strtrim(cellstr(textStrings));  %# Remove any space padding
% [x,y] = meshgrid(1:size(mat,2),1:size(mat,1));   %# Create x and y coordinates for the strings
% midValue = mean(get(gca,'CLim'));  %# Get the middle value of the color range
% textColors = repmat(mat(:) < midValue,1,3);  %# Choose white or black for the
                                             %#   text color of the strings so
                                             %#   they can be easily seen over
                                             %#   the background color

% textColors = repmat(mat(:) < 0.01,1,3).*repmat([1 1 1],size(mat,1)*size(mat,2),1);

% set(hStrings,{'Color'},num2cell(textColors,2));  %# Change the text colors

% if runs == 5
    Y_label = [];
    X_label = {'S. cerevesiae','K. aerogenes','HEK','HeLa','S typhimurium',...
        'E coli (MG1655)','H Salinarium','Contaminated Milk','Uncontaminated Milk'}  ;  
% elseif runs == 7
%     X_label = {'TC','TC','TC','TC','TC','TC','TC','TC','TC','TC',...
%         'RTC','RTC','RTC','RTC','RTC','RTC','RTC','RTC','RTC','RTC',...
%         'RC'};
%     Y_label = {'\lambda_1','\lambda_2','\lambda_3','\lambda_4','\lambda_5','\lambda_6','\lambda_7'}  ;  
% end
% X_label = {'TC_{avg}','RTC_{avg}','RC'};


set(gca,'XTick',1:size(mat,2),...                         %# Change the axes tick marks
        'XTickLabel',[X_label],'XtickLabelRotation',25,...  %#   and tick labels
        'YTick',1:size(mat,1),...
        'YTickLabel',[Y_label],...
        'TickLength',[0 0],'Fontsize',20,'fontweight','bold');

    
correctY=correctY'
[~,index] = max(prob_loo,[],2)
mean(index==correctY)
for i=1:9
    acc(i)=mean(correctY(correctY==i)==index(correctY==i))
end

%%
addpath(genpath('/Users/Mehraveh/Documents/MATLAB/'))
% clear all; close all; clc
Salmonella_data = csvread('/Users/Mehraveh/Documents/MATLAB/Reza/Salmonella/Species101017_clean_PTMs.csv');
[~,j]=ismember(Salmonella_data(:,1),[8,9])
Salmonella_data(j==0,:)=[];
addpath(genpath('/Users/Mehraveh/Documents/MATLAB/Parcellation/2017'))
% l = 174
l = size(Salmonella_data,1)
V = Salmonella_data(1:l,2:end)';
% load label_but_col.mat
load imp_feature.mat
load Acc.mat


[sorted_imp, idx] = sort(imp,'descend')
% label = label(idx,:)
A = V(idx(1:2),:)
[coeff,score,latent,tsquared,explained] = pca(A');
score=score';
%%
% recall = false negative, precision = false positive
close all
figure
cmap = othercolor('Spectral4',length(precision))
for i = 1:size(V,2)    
    s=scatter(A(1,i),A(2,i),500,'MarkerFaceColor',cmap(Salmonella_data(i,1)-min(Salmonella_data(:,1))+1,:),'MarkerEdgeColor','none');, hold on          
    alpha(s,0.8)
%     legendInfo{i} = ['X = ' num2str(i)];
end

clear h
for i = 1:length(precision)
    h(i) = plot(NaN,NaN,'.','color',cmap(i,:),'MarkerSize',50), hold on  
end
% legend('1','2')
samples = {'S. cerevesiae','K. aerogenes', 'HEK', 'HeLa', 'S typhimurium',...
    'E coli', 'H Salinarium','Contaminated Milk', 'Uncontaminated Milk', ...
    'Uncontaminated Milk-Butros', 'Listeria monocytogenes Spiked Milk 10^5-Butros',...
    'Ecoli Spiked Milk 10^4-Butros', 'Klebsiella aerogenes Spiked Milk 10^5-Butros',...
    'Ecoli Spiked Milk 10^5-Butros'}

samples={'Uncontaminated Milk-Butros', 'Listeria monocytogenes Spiked Milk 10^5-Butros',...
    'Ecoli Spiked Milk 10^4-Butros', 'Klebsiella aerogenes Spiked Milk 10^5-Butros',...
    'Ecoli Spiked Milk 10^5-Butros'}


samples={'Contaminated Milk','Uncontaminated Milk'}
legend(h,samples');

% legend(legendInfo)
set(gca,'box','off')
axis('square')
set(gca,'Ticklength', [0 0])
set(gca,'FontSize',25); 
% set(gca,'xticklabelrotation',45); 
set(gca,'FontWeight','normal');
color = get(gca,'Color');
grid on

xlabel(sprintf('First important PTM (%s)',label(1,find(~isspace(label(1,:))))));
ylabel(sprintf('Second important PTM (%s)',label(2,find(~isspace(label(2,:))))));
xlim([0.035,0.08])
ylim([0.65,0.875])
% yruler = ax1.YRuler;
% yruler.Axle.Visible = 'off';
% xruler = ax1.XRuler;
% xruler.Axle.Visible = 'off';
set(gca,'color','none')

% [~, I] = unique(C); %
% I = length(C) - I(length(I):-1:1); 
% p = findobj(gca,'Type','Patch');
% legend(p(I),'X','Y','Z');

%%
figure
accs = [precision;recall]'
cmap = othercolor('Spectral4',length(precision))
% b = bar(accs,'LineWidth',1.5,'FaceColor',[0 .5 .5],'EdgeColor',[0 .9 .9],'LineWidth',1.5)
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
axis square
set(gca,'XTick',0:1:length(precision)-1)
set(gca,'XTickLabelRotation',0)
set(gca,'XTickLabel',samples)

set(gca,'box','off')
set(gca,'Ticklength', [0 0])
set(gca,'FontSize',30); 
set(gca,'FontWeight','normal');
color = get(gca,'Color');
grid on

% ylim([0.5,1.05])
ylabel('Identification Accuracy') 

%%
load label.mat
load imp_feature.mat
load Acc.mat
[sorted_imp, idx] = sort(imp,'descend')
label = label(idx,:)
for i=1:length(label)    
    label_cell{i} = label(i,find(~isspace(label(i,:))));
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




%% exemplar

V(isnan(V)) = 0;
t = size(V,1);
n = size(V,2);
mean_subtract = mean(V,2);
V = V - repmat(mean_subtract,[1,n]);   
twoNorm = sqrt(sum(abs(V).^2,1));
% m = max(twoNorm);
% V = V/m;
V = V./twoNorm;

sqDistances_HCP{1} = sqDistance(V);


imagesc(1-sqDistances_HCP{1});

D = sqDistances_HCP{1};
e0 = zeros(t,1);
e0(1)=e0(1)+3;
d0 = sqDistance_Y(V,e0);

if min(d0)<=max(max(D(1:n,1:n)))
    fprintf('No :( \n')
end       
d0_HCP{1} = d0;
K = max(Salmonella_data(1:l,1));
S_opt = exemplar(sqDistances_HCP,d0_HCP,t,n,K,1);
S_opt = sort(S_opt)
S_opt_group = Salmonella_data(S_opt(1,:),1);

D = sqDistances_HCP{1};
[D_sorted,index_global(:,1)] = min(D(S_opt,1:end),[],1);
index_global(:,2) = Salmonella_data(1:l,1);

for i=1:K
    idx = find(index_global(:,2) == i);
    acc(i,1) = mean(index_global(idx,1) == index_global(idx,2));
end
