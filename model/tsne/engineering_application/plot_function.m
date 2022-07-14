function plot_function(layer, label, dimension, title)
color_select=[[1 0 0]; [0 1 0]; [0 0 1]; [1 1 0]; [1 0 1]; [0 1 1]; [1 0.5 0]; [0.67 0 1]; [0.5 0 0]];
% fold_path="D:\paper_all\second paper\Word\visio\experiment\duanjie1500\results\predict_accuracy\matlab_m\pic_matlab_tsne\";
fold_path='.\pic_matlab_tsne\';
figure(1)

% shape = {'s', 'o', 'o', 'o', '+', '+', 'o', '+', '+'}
% color_list = {1, 2, 3, 4, 5, 6, 7, 8, 9}
% color_select = {[0 0 0]; [1 0 0]; [0 1 0]; [0 0 1]; [1 0 1]; [0 1 1]; [1 0.5 0]; [0.67 0 1]; [0.5 0 0]}
% %  黑, 红， 绿， 蓝， 洋红，青蓝，橘黄，天蓝，深红
% % ['NN', 'NI', 'NO', 'MN', 'MI', 'MO', 'SN', 'SI', 'SO']
% makers = {}
% for j=1:length(color_list)
%     makers=[makers;{shape{j},color_list{j}}];
% end
color_list = {1, 2, 3};
shape={'^', 'o', '+'};

color_select = {[1 0 0]; [0 1 0]; [0 0 1]}; % 红； 绿； 蓝
% ['NN', 'NI', 'NO', 'MN', 'MI', 'MO', 'SN', 'SI', 'SO']
% makers = {1s; 1o; 1+; 2s; 2o; 2+; 3s; 3o; 3+}

makers = {};
for i=1:length(color_list)
    for j=1:length(shape)
            makers=[makers;{shape{j},color_list{i}}];
    end
end

if dimension==2
    x=layer(:,1);
    y=layer(:,2);
    % color_list(1,:)
    for i=1:length(x)
        g = makers(label(1,i)+1,:);
        % scatter(坐标x，坐标y，大小，形状，轮廓线，颜色，线条宽，宽值)
        scatter(x(i,1),y(i,1),80,g{1},'MarkerEdgeColor',color_select{g{2}},'LineWidth',1);

        hold on
    end
    set(gca,'linewidth',1,'fontsize',20,'fontname','Times New Roman');
%     xlabel('(a)','Fontname', 'Times New Roman','FontSize',20);
    b=max(x(:,1));
    a=min(x(:,1));
    d=max(y(:,1));
    c=min(y(:,1));
    set(gca,'XTick',[a(1):(b(1)-a(1))/8:b(1)]);
    set(gca,'YTick',[c(1):(d(1)-c(1))/8:d(1)]);
    set(gca,'XTickLabel',{'', 6, 4, 2, 0, -2, -4, -6,''}, 'Fontname', 'Times New Roman');
    set(gca,'YTickLabel',{'', 6, 4, 2, 0, -2, -4, -6,''}, 'Fontname', 'Times New Roman'); 
    axis([a(1) b(1) c(1) d(1)]);
    box on;
    file_path = strcat(fold_path, title);
    saveas(gcf,file_path,'tif')
else
    x=layer(:,1);
    y=layer(:,2);
    z=layer(:,3);
    % color_list(1,:)
    for i=1:length(x)
        g = makers(label(1,i)+1,:);
        % scatter(坐标x，坐标y，坐标z，大小，形状，轮廓线，颜色，线条宽，宽值)
        scatter3(x(i,1),y(i,1),z(i,1), 80, g{1},'MarkerEdgeColor',color_select{g{2}},'LineWidth',1);

        hold on
    end
    set(gca,'linewidth',1,'fontsize',15,'fontname','Times New Roman');
%     xlabel('(a)','Fontname', 'Times New Roman','FontSize',15);
    b=max(x(:,1));
    a=min(x(:,1));
    d=max(y(:,1));
    c=min(y(:,1));
    e=min(z(:,1));
    f=max(z(:,1));
    set(gca,'XTick',[a(1):(b(1)-a(1))/4:b(1)]+(b(1)-a(1))/8);
    set(gca,'YTick',[c(1):(d(1)-c(1))/4:d(1)]+(d(1)-c(1))/8);
    set(gca,'ZTick',[e(1):(f(1)-e(1))/4:f(1)]+(f(1)-e(1))/8);
    set(gca,'XTickLabel',{-8, -4, 0, 4,8}, 'Fontname', 'Times New Roman');
    set(gca,'YTickLabel',{-8, -4, 0, 4,8}, 'Fontname', 'Times New Roman'); 
    set(gca,'ZTickLabel',{-8, -4, 0, 4,8}, 'Fontname', 'Times New Roman');
    axis([a(1) b(1) c(1) d(1) e(1) f(1)]);
    grid on;
    set(gca,'GridLineStyle',':','GridColor','k','GridAlpha',0.4);
    file_path = strcat(fold_path, title);
    saveas(gcf,file_path,'tif')
end
close all


%{
% 这里是按照9个颜色画图
function plot_function(layer, label, dimension, title)
color_list=[[1 0 0]; [0 1 0]; [0 0 1]; [1 1 0]; [1 0 1]; [0 1 1]; [1 0.5 0]; [0.67 0 1]; [0.5 0 0]];
% fold_path="D:\paper_all\second paper\Word\visio\experiment\duanjie1500\results\predict_accuracy\matlab_m\pic_matlab_tsne\";
fold_path='.\pic_matlab_tsne\';
figure(1)
if dimension==2
    x=layer(:,1);
    y=layer(:,2);
    % color_list(1,:)
    for i=1:length(x)
        color_list(label(1,i)+1,:);
        scatter(x(i,1),y(i,1),8, color_list(label(1,i)+1,:),'filled');
        hold on
    end
    set(gca,'linewidth',1,'fontsize',20,'fontname','Times New Roman');
%     xlabel('(a)','Fontname', 'Times New Roman','FontSize',20);
    b=max(x(:,1));
    a=min(x(:,1));
    d=max(y(:,1));
    c=min(y(:,1));
    set(gca,'XTick',[a(1):(b(1)-a(1))/8:b(1)]);
    set(gca,'YTick',[c(1):(d(1)-c(1))/8:d(1)]);
    set(gca,'XTickLabel',{'', 6, 4, 2, 0, -2, -4, -6,''}, 'Fontname', 'Times New Roman');
    set(gca,'YTickLabel',{'', 6, 4, 2, 0, -2, -4, -6,''}, 'Fontname', 'Times New Roman'); 
    axis([a(1) b(1) c(1) d(1)]);
    box on;
    file_path = strcat(fold_path, title);
    saveas(gcf,file_path,'tif')
else
    x=layer(:,1);
    y=layer(:,2);
    z=layer(:,3);
    % color_list(1,:)
    for i=1:length(x)
        color_list(label(1,i)+1,:);
        scatter3(x(i,1),y(i,1),z(i,1), 8, color_list(label(1,i)+1,:),'filled')
        hold on
    end
    set(gca,'linewidth',1,'fontsize',15,'fontname','Times New Roman');
%     xlabel('(a)','Fontname', 'Times New Roman','FontSize',15);
    b=max(x(:,1));
    a=min(x(:,1));
    d=max(y(:,1));
    c=min(y(:,1));
    e=min(z(:,1));
    f=max(z(:,1));
    set(gca,'XTick',[a(1):(b(1)-a(1))/4:b(1)]+(b(1)-a(1))/8);
    set(gca,'YTick',[c(1):(d(1)-c(1))/4:d(1)]+(d(1)-c(1))/8);
    set(gca,'ZTick',[e(1):(f(1)-e(1))/4:f(1)]+(f(1)-e(1))/8);
    set(gca,'XTickLabel',{-8, -4, 0, 4,8}, 'Fontname', 'Times New Roman');
    set(gca,'YTickLabel',{-8, -4, 0, 4,8}, 'Fontname', 'Times New Roman'); 
    set(gca,'ZTickLabel',{-8, -4, 0, 4,8}, 'Fontname', 'Times New Roman');
    axis([a(1) b(1) c(1) d(1) e(1) f(1)]);
    grid on;
    set(gca,'GridLineStyle',':','GridColor','k','GridAlpha',0.4);
    file_path = strcat(fold_path, title);
    saveas(gcf,file_path,'tif')
end
close all
%}
