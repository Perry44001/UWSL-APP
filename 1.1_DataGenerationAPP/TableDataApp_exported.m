%% Startup
% Read table array from file
% 设置本文件所在路径为当前工作空间路径
filep = mfilename('fullpath'); %filep包含了本m文件所在的路径已经以及文件名（不带.m后缀）
[pathstr,~]=fileparts(filep);%pathstr才是本m文件所在的路径
cd(pathstr);%更改当前活动目录路径
% addpath([pathstr,'\函数']);       % 将辅助函数文件所在位置添加到搜索路径顶端

table1 = readtable('.\1.1.0SettingTables\Parameters.xlsx', 'VariableNamingRule','preserve');

ssp = load('.\1.1.1EnvBaseDataSet\ssp.txt');
rd = [10 20 30 40 50 60];
top_option = 'CVW';

%% LoadSettingExcel
[file,path] =uigetfile('*.xlsx','Loading File','test.xlsx');
if isequal(file,0) || isequal(path,0)
    disp('User clicked Cancel.')
else
    disp(['User selected ',fullfile(path,file),...
        ' .'])
    table1 = readtable( [path '/' file], 'VariableNamingRule','preserve');
end

%% LoadSSP
[file,path] =uigetfile('*.txt','Loading File','test.txt');
if isequal(file,0) || isequal(path,0)
    disp('User clicked Cancel.')
else
    disp(['User selected ',fullfile(path,file),...
        ' .'])
    ssp = load( [path '/' file]);
end

%% RUN
figure;
UIAxes1 = gca;
figure;
UIAxes2 = gca;
FunSWDataSim_exported(table1, ssp, rd, top_option, UIAxes1, UIAxes2);

