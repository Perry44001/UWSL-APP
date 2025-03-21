function FunSWDataSim(table1, menu, UIAxes1, UIAxes2, handles)
% 1. Shallow water environmet
% An overview for shallow water environment sound propagation, including env file construction, flp file construction, sound speed profile visualizing, nomal mode calculation and visualization, acoustic field calculation and transmission loss visualization.
% 1.1. Write env file
ssp = menu.ssp;              % Sound speed profile
if (max(ssp(:,1))<max([table1{5,4}, table1{12,4}]))
    error("SSP depth must extrop the scan sea water depth.");
elseif ssp(1,1) ~= 0
    error("SSP depth must begin at 0 m.");
end
psetpath = uigetdir;
if isequal(psetpath,0)
    disp('User clicked Cancel.')
    return;
end
disp(['User selected ', psetpath, ' .'])
cd(psetpath);
folder='.\(1)NarrowbandEnvfile';                        % Folder name definition
if exist(folder,'dir')==0                               % Determine if a folder exists
    mkdir(folder);                                      % Create a folder when it does not exist
else
    disp('dir is exist');                               % If the folder exists, output :dir is exist
end
cd(folder);                                             % Change the path to the environment file path

envfil = 'dataset';                                     % env file name
model = menu.select_model;                              % env file type
% env file setting
% (1) - Title
TitleEnv = 'Sea environment';                           % env title

% (2) - Frequency
freqvec = linspace(table1{1,3}, table1{1,4}, table1{1,5});% Broadband frequency vector
freq = table1{1,3};                                     % FREQ: Frequency in Hz.

% (3) - Number of Media
SSP.NMedia = 1;                                         % Number of media

% (4) - Top Option
Bdry.Top.Opt = 'CVW';                                   % Top option

% (5) - Sound Speed Profile
% sea water
SSP.N(1) = 0;                                           % NMESH:   Number of mesh points used in the internal discretization.
SSP.sigma(1) = 0;                                       % SIGMA:   RMS roughness at the interface (ignored by BELLHOP and SPARC)
water_depth_vec = linspace(table1{5,3}, table1{5,4}, table1{5,5}); % Depth of the 2nd media (m).
index_c = ceil(length(water_depth_vec)/2);
water_depth_centre = water_depth_vec(index_c);          % 中心点值
SSP.depth(2) = water_depth_centre;                      % Depth of the 2nd media (m).

SSP.raw(1).z = ssp(ssp(:,1)<=SSP.depth(2),1);           % Set ssp depth z():     Depth (m)
if ~ismember(SSP.raw(1).z, SSP.depth(2))                % If max depth is not in z(), we expand z().
    SSP.raw(1).z = [SSP.raw(1).z.' SSP.depth(2)];
end
SSP.Nz(1) = length(SSP.raw(1).z);                       % Number of z vector
SSP.raw(1).alphaR(1:SSP.Nz(1)) = interp1(ssp(:,1), ssp(:,2), SSP.raw(1).z, 'linear'); % alphaR():    P-wave speed (m/s).
SSP.raw(1).betaR(1:SSP.Nz(1)) = 0;                      % betaR(): S-wave speed (m/s).
SSP.raw(1).rho(1:SSP.Nz(1)) = 1.03;                     % rho():   Density (g/cm3).
SSP.raw(1).alphaI(1:SSP.Nz(1)) = 0;                     % alphaI():P-wave attenuation (units as given in Block 2)
SSP.raw(1).betaI(1:SSP.Nz(1)) = 0;                      % betaI(): S-wave attenuation

% (6) - Bottom Option
Bdry.Bot.Opt = 'A';                                     % BOTOPT(1:1): Type of bottom boundary condition.
Basement_alphaR_vec = linspace(table1{6,3}, table1{6,4}, table1{6,5});
index_c = ceil(length(Basement_alphaR_vec)/2);
Basement_alphaR_centre = Basement_alphaR_vec(index_c);
Bdry.Bot.HS.alphaR = Basement_alphaR_centre;            % alphaR:  Bottom P-wave speed (m/s).

Basement_rho_vec = linspace(table1{7,3}, table1{7,4}, table1{7,5});
index_c = ceil(length(Basement_rho_vec)/2);
Basement_rho_centre = Basement_rho_vec(index_c);
Bdry.Bot.HS.rho = Basement_rho_centre;                  % rho:     Bottom density (g/cm3).
Basement_alphaI_vec = linspace(table1{8,3}, table1{8,4}, table1{8,5});
index_c = ceil(length(Basement_alphaI_vec)/2);
Basement_alphaI_centre = Basement_alphaI_vec(index_c);
Bdry.Bot.HS.alphaI = Basement_alphaI_centre;            % alphaI:  Bottom P-wave attenuation. (units as given by TOPOPT(3:3) )
Bdry.Bot.HS.betaR = 0;                                  % betaR:   Bottom S-wave speed (m/s).
Bdry.Bot.HS.betaI = 0;                                  % betaI:   Bottom S-wave attenuation.

% (7) - Phase Speed Limits
cInt.Low = 0;                                           % CLOW:    Lower phase speed limit (m/s).
cInt.High = 2e4;                                        % CHIGH:   Upper phase speed limit (m/s).

% (8) - Maximum Range
RMax = 0;                                               % RMax:    Maximum range (km)

% (9) - Source/Receiver Depths (声场互易)
Pos.r.range = linspace(table1{3,3}, table1{3,4}, table1{3,5});  % The range distances between sources and receivers (km)
Pos.r.z = linspace(table1{4,3}, table1{4,4}, table1{4,5});      % Sz(): Source   depths (m).
Pos.s.z = menu.rd;

% (10) - Beam (BELLHOP)
Beam.RunType = 'CB';                                              % Not used in kraken.
Beam.Nbeams = 0;                                                  % Number of beams.
Beam.alpha = [-89 89];                                            % 掠射角 °
Beam.deltas = 0;                                                  % 声线步长
Beam.Box.r = max(Pos.r.range)+1;                                  % Box距离
Beam.Box.z = max(water_depth_vec)+10;                             % Box深度

% flp file setting
%  (2) - OPTIONS
Option = 'RA'; % 'R' point source. 'A' Adiabatic mode theory (default).

% 1.3. Plot sound speed profile
% figure;
plotsspUI(UIAxes2, SSP, Bdry.Top.Opt(1))
% uiplotssp(UIAxes2, envfil)

% 1.4. Plot nomal mode vs depth and 2 dimension transmission loss vs range and depth
global units;
units = 'km';           % Change the range units from m to km
if strcmp(model, 'KRAKEN')
    write_env( envfil, model, TitleEnv, freq, SSP, Bdry, Pos, Beam, cInt, RMax );
    write_fieldflp( envfil, Option, Pos );
    kraken(envfil);          % Run kraken
elseif strcmp(model, 'BELLHOP')
    write_env( envfil, model, TitleEnv, freq, SSP, Bdry, Pos, Beam, cInt, RMax );
    bellhop(envfil);         % Run bellhop
    % shd2mat([envfil '.shd']);
elseif strcmp(model, 'RAM')
    write_env_RAM( envfil, model, TitleEnv, freq, SSP, Bdry, Pos);
    ram_p(envfil);           % Run ram_p
    grid2mat([envfil '.grid'], [envfil '.shd.mat']);
end
uiplotshd(UIAxes1, [envfil '.shd'])

% 2. Dataset generation
% 2.1. Broadband env and flp file for training dataset
folder='..\(2)BroadbandEnvfile';
if exist(folder, 'dir')==0          % Determine if a folder exists
    mkdir(folder);                  % Create a folder when it does not exist
else
    disp('dir is exist');           % If the folder exists, output :dir is exist
end
cd(folder);                         % Change the path to the environment file path

folder_write_b_vec = {'A.TrainingSet', 'B.TestSet'};
% Bdry.Top.Opt = 'CVF  B';          % Top option
Pos.freqvec = linspace(table1{1,3}, table1{1,4}, table1{1,5});% Broadband frequency vector

envfil_type_list = {'Training', 'Test'}; % env file name
for itype = 1: length(envfil_type_list)
    Pos.r.range = linspace(table1{3+(itype-1)*7,3}, table1{3+(itype-1)*7,4}, table1{3+(itype-1)*7,5}); % Rr(): ranges (m). 声场互易
    Pos.r.z = linspace(table1{4+(itype-1)*7,3}, table1{4+(itype-1)*7,4}, table1{4+(itype-1)*7,5});     % Rz(): Source depths (m). 声场互易
    % 四个变化量，目前预设的是 海水深度、底质纵波速度、底质密度、底质纵波衰减
    % 海水深度
    water_depth_vec = linspace(table1{5+(itype-1)*7,3}, table1{5+(itype-1)*7,4}, table1{5+(itype-1)*7,5}); % Depth of the 2nd media (m).
    % 底质纵波速度
    Basement_alphaR_vec = linspace(table1{6+(itype-1)*7,3}, table1{6+(itype-1)*7,4}, table1{6+(itype-1)*7,5});
    % 底质密度
    Basement_rho_vec = linspace(table1{7+(itype-1)*7,3}, table1{7+(itype-1)*7,4}, table1{7+(itype-1)*7,5});
    % 底质纵波衰减
    Basement_alphaI_vec = linspace(table1{8+(itype-1)*7,3}, table1{8+(itype-1)*7,4}, table1{8+(itype-1)*7,5});
    env_list = [];
    mat_list = [];
    for ifreq = 1: length(freqvec)
        freq = freqvec(ifreq);
        % 四个变化量的中心索引
        iw_c = ceil(length(water_depth_vec)/2);
        ibar_c = ceil(length(Basement_alphaR_vec)/2);
        ibr_c = ceil(length(Basement_rho_vec)/2);
        ibai_c = ceil(length(Basement_alphaI_vec)/2);
        % 海水深度变化
        for iw = 1: length(water_depth_vec)
            % 海水深度变化需要对声速剖面进行插值，目前使用的是线性外插法
            SSP.depth(2) = water_depth_vec(iw);
            SSP.raw(1).z = ssp(ssp(:,1)<=SSP.depth(2),1);           % Set ssp depth z():     Depth (m)

            if ~ismember(SSP.raw(1).z, SSP.depth(2))                % If max depth is not in z(), we expand z().
                SSP.raw(1).z = [SSP.raw(1).z.' SSP.depth(2)];
            end
            SSP.Nz(1) = length(SSP.raw(1).z);                       % Number of z vector
            SSP.raw(1).alphaR(1:SSP.Nz(1)) = interp1(ssp(:,1), ssp(:,2), SSP.raw(1).z, 'linear'); % alphaR():    P-wave speed (m/s).
            SSP.raw(1).betaR(1:SSP.Nz(1)) = 0;                      % betaR(): S-wave speed (m/s).
            SSP.raw(1).rho(1:SSP.Nz(1)) = 1.03;                     % rho():   Density (g/cm3).
            SSP.raw(1).alphaI(1:SSP.Nz(1)) = 0;                     % alphaI():P-wave attenuation (units as given in Block 2)
            SSP.raw(1).betaI(1:SSP.Nz(1)) = 0;                      % betaI(): S-wave attenuation
            envfil = sprintf('%s_iw_%02d_ibar_%02d_ibr_%02d_ibai_%02d_ifreq_%03d', envfil_type_list{itype}, iw, ibar_c, ibr_c, ibai_c, ifreq);
            env_list = [env_list; envfil];
            % 写环境文件 env flp 和 in
            if strcmp(model, 'KRAKEN')
                write_env( envfil, model, TitleEnv, freq, SSP, Bdry, Pos, Beam, cInt, RMax );
                write_fieldflp( envfil, Option, Pos );
            elseif strcmp(model, 'BELLHOP')
                write_env( envfil, model, TitleEnv, freq, SSP, Bdry, Pos, Beam, cInt, RMax );
            elseif strcmp(model, 'RAM')
                write_env_RAM( envfil, model, TitleEnv, freq, SSP, Bdry, Pos);
            end
        end
        % 写完环境参数后，需要将参数改回为中心参数
        SSP.depth(2) = water_depth_centre;
        SSP.raw(1).z = ssp(ssp(:,1)<=SSP.depth(2),1);           % Set ssp depth z():     Depth (m)

        if ~ismember(SSP.raw(1).z, SSP.depth(2))                % If max depth is not in z(), we expand z().
            SSP.raw(1).z = [SSP.raw(1).z.' SSP.depth(2)];
        end
        SSP.Nz(1) = length(SSP.raw(1).z);                       % Number of z vector
        SSP.raw(1).alphaR(1:SSP.Nz(1)) = interp1(ssp(:,1), ssp(:,2), SSP.raw(1).z, 'linear'); % alphaR():    P-wave speed (m/s).
        SSP.raw(1).betaR(1:SSP.Nz(1)) = 0;                      % betaR(): S-wave speed (m/s).
        SSP.raw(1).rho(1:SSP.Nz(1)) = 1.03;                     % rho():   Density (g/cm3).
        SSP.raw(1).alphaI(1:SSP.Nz(1)) = 0;                     % alphaI():P-wave attenuation (units as given in Block 2)
        SSP.raw(1).betaI(1:SSP.Nz(1)) = 0;                      % betaI(): S-wave attenuation

        % 底质纵波速度变化
        for ibar = 1: length(Basement_alphaR_vec)
            Bdry.Bot.HS.alphaR = Basement_alphaR_vec(ibar);            % alphaR:  Bottom P-wave speed (m/s).
            envfil = sprintf('%s_iw_%02d_ibar_%02d_ibr_%02d_ibai_%02d_ifreq_%03d', envfil_type_list{itype}, iw_c, ibar, ibr_c, ibai_c, ifreq);
            env_list = [env_list; envfil];
            % 写环境文件 env flp 和 in
            if strcmp(model, 'KRAKEN')
                write_env( envfil, model, TitleEnv, freq, SSP, Bdry, Pos, Beam, cInt, RMax );
                write_fieldflp( envfil, Option, Pos );
            elseif strcmp(model, 'BELLHOP')
                write_env( envfil, model, TitleEnv, freq, SSP, Bdry, Pos, Beam, cInt, RMax );
            elseif strcmp(model, 'RAM')
                write_env_RAM( envfil, model, TitleEnv, freq, SSP, Bdry, Pos);
            end
        end
        Bdry.Bot.HS.alphaR = Basement_alphaR_centre;

        % 底质密度变化
        for ibr = 1: length(Basement_rho_vec)
            Bdry.Bot.HS.rho = Basement_rho_vec(ibr);                  % rho:     Bottom density (g/cm3).
            envfil = sprintf('%s_iw_%02d_ibar_%02d_ibr_%02d_ibai_%02d_ifreq_%03d', envfil_type_list{itype}, iw_c, ibar_c, ibr, ibai_c, ifreq);
            env_list = [env_list; envfil];
            % 写环境文件 env flp 和 in
            if strcmp(model, 'KRAKEN')
                write_env( envfil, model, TitleEnv, freq, SSP, Bdry, Pos, Beam, cInt, RMax );
                write_fieldflp( envfil, Option, Pos );
            elseif strcmp(model, 'BELLHOP')
                write_env( envfil, model, TitleEnv, freq, SSP, Bdry, Pos, Beam, cInt, RMax );
            elseif strcmp(model, 'RAM')
                write_env_RAM( envfil, model, TitleEnv, freq, SSP, Bdry, Pos);
            end
        end
        Bdry.Bot.HS.rho = Basement_rho_centre;

        % 底质纵波衰减变化
        for ibai = 1: length(Basement_alphaI_vec)
            Bdry.Bot.HS.alphaI = Basement_alphaI_vec(ibai);            % alphaI:  Bottom P-wave attenuation. (units as given by TOPOPT(3:3) )
            envfil = sprintf('%s_iw_%02d_ibar_%02d_ibr_%02d_ibai_%02d_ifreq_%03d', envfil_type_list{itype}, iw_c, ibar_c, ibr_c, ibai, ifreq);
            env_list = [env_list; envfil];
            % 写环境文件 env flp 和 in
            if strcmp(model, 'KRAKEN')
                write_env( envfil, model, TitleEnv, freq, SSP, Bdry, Pos, Beam, cInt, RMax );
                write_fieldflp( envfil, Option, Pos );
            elseif strcmp(model, 'BELLHOP')
                write_env( envfil, model, TitleEnv, freq, SSP, Bdry, Pos, Beam, cInt, RMax );
            elseif strcmp(model, 'RAM')
                write_env_RAM( envfil, model, TitleEnv, freq, SSP, Bdry, Pos);
            end
        end
        Bdry.Bot.HS.alphaI = Basement_alphaI_centre;
    end

    folder = ['..\' folder_write_b_vec{itype}];
    if exist(folder, 'dir')==0 % Determine if a folder exists
        mkdir(folder);  % Create a folder when it does not exist
    else
        disp('dir is exist'); % If the folder exists, output :dir is exist
    end
    % run at
    if strcmp(model, 'KRAKEN')
        parfor i = 1: size(env_list,1)
            envfil = env_list(i,:);
            krakenc(envfil);          % Run kraken
        end
    elseif strcmp(model, 'BELLHOP')
        parfor i = 1: size(env_list,1)
            envfil = env_list(i,:);
            bellhopcxx(envfil);          % Run kraken
        end
    elseif strcmp(model, 'RAM')
        parfor i = 1: size(env_list,1)
            envfil = env_list(i,:);
            ram_p(envfil);          % Run kraken
        end
    end

    for iw = 1: length(water_depth_vec)
        pressure = zeros(length(freqvec), length(Pos.s.z), length(Pos.r.z), length(Pos.r.range));
        for ifreq = 1: length(freqvec)
            envfil = sprintf('%s_iw_%02d_ibar_%02d_ibr_%02d_ibai_%02d_ifreq_%03d', envfil_type_list{itype}, iw, ibar_c, ibr_c, ibai_c, ifreq);
            if strcmp(model, 'KRAKEN') || strcmp(model, 'BELLHOP')
                [ ~, ~, ~, ~, ~, ~, pres ] = read_shd([envfil '.shd']);
                pressure(ifreq, :, :, :) = pres(1,  :, :, :);
            elseif strcmp(model, 'RAM')
                pres = read_grid([envfil '.grid']);
                pressure(ifreq, :, :, :) = pres;
            end
        end
        matfil = [envfil(1:end-10) '.mat'];
        mat_list = [mat_list; matfil];
        save(matfil, 'pressure');
    end

    if length(Basement_alphaR_vec)>1 % 如果只有一个的话，在第一个循环已经保存过了
        for ibar = 1: length(Basement_alphaR_vec)
            pressure = zeros(length(freqvec), length(Pos.s.z), length(Pos.r.z), length(Pos.r.range));
            for ifreq = 1: length(freqvec)
                envfil = sprintf('%s_iw_%02d_ibar_%02d_ibr_%02d_ibai_%02d_ifreq_%03d', envfil_type_list{itype}, iw_c, ibar, ibr_c, ibai_c, ifreq);
                if strcmp(model, 'KRAKEN') || strcmp(model, 'BELLHOP')
                    [ ~, ~, ~, ~, ~, ~, pres ] = read_shd([envfil '.shd']);
                    pressure(ifreq, :, :, :) = pres(1,  :, :, :);
                elseif strcmp(model, 'RAM')
                    pres = read_grid([envfil '.grid']);
                    pressure = cat(1, pressure, pres);
                end
            end
            matfil = [envfil(1:end-10) '.mat'];
            mat_list = [mat_list; matfil];
            save(matfil, 'pressure');
        end
    end
    if length(Basement_rho_vec)>1 % 如果只有一个的话，在第一个循环已经保存过了
        for ibr = 1: length(Basement_rho_vec)
            pressure = zeros(length(freqvec), length(Pos.s.z), length(Pos.r.z), length(Pos.r.range));
            for ifreq = 1: length(freqvec)
                envfil = sprintf('%s_iw_%02d_ibar_%02d_ibr_%02d_ibai_%02d_ifreq_%03d', envfil_type_list{itype}, iw_c, ibar_c, ibr, ibai_c, ifreq);
                if strcmp(model, 'KRAKEN') || strcmp(model, 'BELLHOP')
                    [ ~, ~, ~, ~, ~, ~, pres ] = read_shd([envfil '.shd']);
                    pressure(ifreq, :, :, :) = pres(1,  :, :, :);
                elseif strcmp(model, 'RAM')
                    pres = read_grid([envfil '.grid']);
                    pressure = cat(1, pressure, pres);
                end
            end
            matfil = [envfil(1:end-10) '.mat'];
            mat_list = [mat_list; matfil];
            save(matfil, 'pressure');
        end
    end
    if length(Basement_alphaI_vec)>1 % 如果只有一个的话，在第一个循环已经保存过了
        for ibai = 1: length(Basement_alphaI_vec)
            pressure = zeros(length(freqvec), length(Pos.s.z), length(Pos.r.z), length(Pos.r.range));
            for ifreq = 1: length(freqvec)
                envfil = sprintf('%s_iw_%02d_ibar_%02d_ibr_%02d_ibai_%02d_ifreq_%03d', envfil_type_list{itype}, iw_c, ibar_c, ibr_c, ibai, ifreq);
                if strcmp(model, 'KRAKEN') || strcmp(model, 'BELLHOP')
                    [ ~, ~, ~, ~, ~, ~, pres ] = read_shd([envfil '.shd']);
                    pressure(ifreq, :, :, :) = pres(1,  :, :, :);
                elseif strcmp(model, 'RAM')
                    pres = read_grid([envfil '.grid']);
                    pressure = cat(1, pressure, pres);
                end
            end
            matfil = [envfil(1:end-10) '.mat'];
            mat_list = [mat_list; matfil];
            save(matfil, 'pressure');
        end
    end

    for i = 1: size(mat_list,1)
        shdmat_fil = mat_list(i, :);

        % 2.3. Write to binary file
        load(shdmat_fil, 'pressure'); % Load data from .mat
        [~, shdmat_fil, ~] = fileparts(shdmat_fil); % 去掉后缀
        filepath = sprintf('%s\\%s.sim', folder, shdmat_fil); % Simulation dataset file name
        fid = fopen(filepath,'wb+'); % Create a binary file
        % Write pressure to the binary file
        for isd = 1: length(Pos.r.z)                        % Source depth circulation
            for isr = 1: length(Pos.r.range)                % Source range circulation
                for ifreq = 1 :length(Pos.freqvec)          % Source frequency circulation
                    pr = reshape(squeeze(pressure(ifreq,:,isd,isr)).',[],1); % Take out the array received sound pressure from the data
                    pr2 = sqrt(pr' * pr);
                    if pr2 == 0
                        pr2 = 1;
                    end
                    pr_nor = pr/ pr2;            % N2 normalized
                    fwrite(fid, real(pr_nor),'float32');    % Write real part pressure
                    fwrite(fid, imag(pr_nor),'float32');    % Write image part pressure
                end
                fwrite(fid, Pos.r.range(isr),'float32');    % Write Source range
                fwrite(fid, Pos.r.z(isd),'float32');        % Write Source depth
            end
        end

        fclose(fid); % Close the binary file
        fclose all;
    end
    handle = handles(itype);
    handle.Value = sprintf('%s simulation: %d%%',folder_write_b_vec{itype}, 100);
    drawnow
    % Check data size
    envfil = sprintf('%s_%03d', envfil_type_list{itype}, 1);
    filepath = sprintf('%s\\*.sim', folder); % Simulation dataset file name
    D = dir(filepath);

    if abs(D(end).bytes- (length(Pos.r.z) *length(Pos.r.range) *(length(Pos.freqvec)* 2*length(Pos.s.z)+2))* 4)< 1e-3
        fprintf('******************\n\n')
        fprintf('Pass the check!\n\n')
        fprintf('******************\n');
    else
        fprintf('******************\n\n')
        fprintf('Failure to pass the data check! Please check the code.\n\n')
        fprintf('******************\n\n')
    end
end

%% 写config

% 初始化结构体
jsonStruct = struct();
param = table1{1,1}{1};
unit = table1{1,2}{1};      % 单位（假设Excel中列名为Unit）
lowerLimit = table1{1,3};   % 下限（假设Excel中列名为LowerLimit）
upperLimit = table1{1,4};   % 上限（假设Excel中列名为UpperLimit）
numValues = table1{1,5};    % 数值个数（假设Excel中列名为NumValues）

% 将数据添加到结构体中
jsonStruct.(param) = struct(...
    'Unit', unit, ...
    'LowerLimit', lowerLimit, ...
    'UpperLimit', upperLimit, ...
    'NumValues', numValues ...
    );

% 遍历每一行并填充结构体
for i = 3:8
    param = table1{i,1}{1};     % 参数名称（假设Excel中列名为Parameter）
    unit = table1{i,2}{1};      % 单位（假设Excel中列名为Unit）
    lowerLimit = table1{i,3};   % 下限（假设Excel中列名为LowerLimit）
    upperLimit = table1{i,4};   % 上限（假设Excel中列名为UpperLimit）
    numValues = table1{i,5};    % 数值个数（假设Excel中列名为NumValues）

    % 将数据添加到结构体中
    jsonStruct.trainset.(param) = struct(...
        'Unit', unit, ...
        'LowerLimit', lowerLimit, ...
        'UpperLimit', upperLimit, ...
        'NumValues', numValues ...
        );
end

for i = 10:15
    param = table1{i,1}{1};     % 参数名称（假设Excel中列名为Parameter）
    unit = table1{i,2}{1};      % 单位（假设Excel中列名为Unit）
    lowerLimit = table1{i,3};   % 下限（假设Excel中列名为LowerLimit）
    upperLimit = table1{i,4};   % 上限（假设Excel中列名为UpperLimit）
    numValues = table1{i,5};    % 数值个数（假设Excel中列名为NumValues）

    % 将数据添加到结构体中
    jsonStruct.testset.(param) = struct(...
        'Unit', unit, ...
        'LowerLimit', lowerLimit, ...
        'UpperLimit', upperLimit, ...
        'NumValues', numValues ...
        );
end

% 添加接收深度和声速剖面（假设这些数据是固定的）
jsonStruct.ReceiverDepth = Pos.s.z;
jsonStruct.SoundSpeedProfile = ssp;

% 将结构体转换为JSON字符串
jsonStr = jsonencode(jsonStruct, 'PrettyPrint', true);

% 将JSON字符串写入文件
fid = fopen(sprintf("%s/config.json", psetpath), 'wt');
fprintf(fid, '%s', jsonStr);
fclose(fid);


cd('..') % Change the path back
end
