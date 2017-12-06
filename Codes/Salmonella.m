clc
clear all
close all
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
