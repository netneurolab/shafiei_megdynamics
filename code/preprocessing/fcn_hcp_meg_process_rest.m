function fcn_hcp_meg_process_rest(hcp_dir, subjList, badChannels, reports_dir)
% hcp_meg_process: Script to process Resting-state MEG data from Human
% Connectime Project
%
% Modified from Brainstorm online tutorials:
%     https://neuroimage.usc.edu/brainstorm/Tutorials/HCP-MEG
%
% Inputs:
%     hcp_dir: Directory with subject subdirectories that have the unzipped
%     HCP files
%     subjList: List of subjects in a Matlab cell format (e.g. {'102816'})
%     badChannels: List of bad channels for each subject in a Matlab cell
%     fromat (e.g. {{'A2', 'A237', 'A244', 'A246', 'A8'}, {'A126', 'A2',
%     'A244', 'A246'}})
%     reports_dir: Directory to save Brainstorm generated reports for each
%     subject
%
% Outputs:
%     saves vertex-level source time-series, SNR, and power for each
%     subject under fullfile(hcp_dir, 'brainstormResults')

% @=============================================================================
% This code is a part of MEG timeseries analysis research project and it is
% still under development.
% =============================================================================@
%
% Author: Golia Shafiei, 2022

%% ===== SCRIPT VARIABLES =====
% Full list of subjects to process, bad channels
SubjectNames = subjList;
BadChannels = badChannels;

%% ===== CREATE PROTOCOL =====
% The protocol name has to be a valid folder name (no spaces, no weird characters...)
ProtocolName = 'megHCP';
% Start brainstorm without the GUI
if ~brainstorm('status')
    brainstorm nogui
    %brainstorm server
end
% Delete existing protocol
gui_brainstorm('DeleteProtocol', ProtocolName);
% Create new protocol
gui_brainstorm('CreateProtocol', ProtocolName, 0, 0);

%% ===== FILES TO IMPORT =====
% You have to specify the folder in which the tutorial dataset is unzipped
if (nargin < 3) || isempty(hcp_dir) || ~file_exist(hcp_dir)
    error('The first argument must be the full path to the tutorial dataset folder.');
end
% Output folder for reports
if (nargin < 3) || isempty(reports_dir) || ~isdir(reports_dir)
    reports_dir = [];
end

%% ===== PRE-PROCESS AND IMPORT =====
for iSubj = 1:length(SubjectNames)
    tic
    % Start a new report (one report per subject)
    bst_report('Start');
    fprintf('\n===== IMPORT: SUBJECT #%d =====\n', iSubj);

    % If subject already exists: delete it
    [sSubject, iSubject] = bst_get('Subject', SubjectNames{iSubj});
    if ~isempty(sSubject)
        db_delete_subjects(iSubject);
    end

    % ===== FILES TO IMPORT =====
    % Build the path of the files to import
    AnatDir    = fullfile(hcp_dir, SubjectNames{iSubj}, 'MEG', 'anatomy');
    Run1File   = fullfile(hcp_dir, SubjectNames{iSubj}, 'unprocessed', 'MEG', '3-Restin', '4D', 'c,rfDC');
    NoiseFile  = fullfile(hcp_dir, SubjectNames{iSubj}, 'unprocessed', 'MEG', '1-Rnoise', '4D', 'c,rfDC');
    % Check if the folder contains the required files
    if ~file_exist(AnatDir) || ~file_exist(Run1File) || ~file_exist(NoiseFile)
        error(['The folder ' hcp_dir ' does not contain subject #' SubjectNames{iSubj} 'from the HCP-MEG distribution.']);
    end

    % ===== IMPORT DATA =====
    % Process: Import anatomy folder
    bst_process('CallProcess', 'process_import_anatomy', [], [], ...
        'subjectname', SubjectNames{iSubj}, ...
        'mrifile',     {AnatDir, 'HCPv3'}, ...
        'nvertices',   15000);

    % Process: Create link to raw files
    sFilesRun1 = bst_process('CallProcess', 'process_import_data_raw', [], [], ...
        'subjectname',  SubjectNames{iSubj}, ...
        'datafile',     {Run1File, '4D'}, ...
        'channelalign', 1);
    sFilesNoise = bst_process('CallProcess', 'process_import_data_raw', [], [], ...
        'subjectname',  SubjectNames{iSubj}, ...
        'datafile',     {NoiseFile, '4D'}, ...
        'channelalign', 1);
    sFilesRaw = [sFilesRun1, sFilesNoise];

    % Process: Resample: 508.63Hz
    sFilesResamp = bst_process('CallProcess', 'process_resample', sFilesRaw, [], ...
        'freq',     508.6275, ...
        'read_all', 1);

    % ===== PRE-PROCESSING =====
    % Process: Notch filter: 60Hz 120Hz 180Hz 240Hz 300Hz
    sFilesNotch = bst_process('CallProcess', 'process_notch', sFilesResamp, [], ... %sFilesRaw
        'freqlist',    [60, 120, 180, 240, 300], ...
        'sensortypes', 'MEG, EEG', ...
        'read_all',    1);

    % Process: High-pass:0.3Hz
    sFilesBand = bst_process('CallProcess', 'process_bandpass', sFilesNotch, [], ...
        'sensortypes', 'MEG, EEG', ...
        'highpass',    0.3, ...
        'lowpass',     0, ...
        'attenuation', 'strict', ...  % 60dB
        'mirror',      0, ...
        'useold',      0, ...
        'read_all',    1);

    % Process: Power spectrum density (Welch)
    sFilesPsdAfter = bst_process('CallProcess', 'process_psd', sFilesBand, [], ...
        'timewindow',  [], ...
        'win_length',  4, ...
        'win_overlap', 50, ...
        'sensortypes', 'MEG, EEG', ...
        'edit',        struct(...
             'Comment',         'Power', ...
             'TimeBands',       [], ...
             'Freqs',           [], ...
             'ClusterFuncTime', 'none', ...
             'Measure',         'power', ...
             'Output',          'all', ...
             'SaveKernel',      0));

    % Mark bad channels
    bst_process('CallProcess', 'process_channel_setbad', sFilesBand, [], ...
                'sensortypes', BadChannels{iSubj});

    % Process: Snapshot: Frequency spectrum
    bst_process('CallProcess', 'process_snapshot', sFilesPsdAfter, [], ...
        'target',         10, ...  % Frequency spectrum
        'modality',       1);      % MEG (All)

    % Process: Delete folders
    bst_process('CallProcess', 'process_delete', ...
        [sFilesRaw, sFilesNotch, sFilesResamp], [], ...
        'target', 2);  % Delete folders

    % ===== ARTIFACT CLEANING =====
    % Process: Select data files in: */*
    sFilesBand = bst_process('CallProcess', 'process_select_files_data', [], [], ...
        'subjectname', SubjectNames{iSubj});

    % Process: Select file names with tag: 3-Restin
    sFilesRest = bst_process('CallProcess', 'process_select_tag', sFilesBand, [], ...
        'tag',    '3-Restin', ...
        'search', 1, ...  % Search the file names
        'select', 1);  % Select only the files with the tag

    % Process: Detect heartbeats
    bst_process('CallProcess', 'process_evt_detect_ecg', sFilesRest, [], ...
        'channelname', 'ECG+, -ECG-', ...
        'timewindow',  [], ...
        'eventname',   'cardiac');

    % Process: Detect eye blinks
    bst_process('CallProcess', 'process_evt_detect_eog', sFilesRest, [], ...
        'channelname', 'VEOG+, -VEOG-', ...
        'timewindow', [], ...
        'eventname', 'blink');

    % Process: Remove simultaneous (keep blinks over heart beats)
    bst_process('CallProcess', 'process_evt_remove_simult', sFilesRest, [], ...
        'remove', 'cardiac', ...
        'target', 'blink', ...
        'dt', 0.25, ...
        'rename', 0);

    % Process: SSP ECG: cardiac (force remove 1st component)
    bst_process('CallProcess', 'process_ssp_ecg', sFilesRest, [], ...
        'eventname',   'cardiac', ...
        'sensortypes', 'MEG', ...
        'usessp',      1, ...
        'select',      1);

    % Process: SSP EOG: blink (force remove 1st component)
    bst_process('CallProcess', 'process_ssp_eog', sFilesRest, [], ...
        'eventname', 'blink', ...
        'sensortypes', 'MEG', ...
        'usessp', 1, ...
        'select', 1);

    % SSP: Noisy signal, Sacades, EMG
    % Process: Detect other artifacts (mark noisy segments)
    bst_process('CallProcess', 'process_evt_detect_badsegment', ...
        sFilesRest, [], ...
        'timewindow', [], ...
        'sensortypes', 'MEG, EEG', ...
        'threshold', 3, ...  % 3
        'isLowFreq', 1, ...
        'isHighFreq', 1);

    % Process: SSP for low frequencies (saccades) 1 - 7 Hz (remove 1st)
    bst_process('CallProcess', 'process_ssp', sFilesRest, [], ...
        'timewindow',  [], ...
        'eventname',   '1-7Hz', ...
        'eventtime',   [], ...
        'bandpass',    [1.5, 7], ...
        'sensortypes', 'MEG', ...
        'usessp',      1, ...
        'saveerp',     0, ...
        'method',      1, ...  % PCA: One component per sensor
        'select',      1);

    % Process: SSP for high frequencies (muscle) 40 - 240 Hz (remove 1st)
    bst_process('CallProcess', 'process_ssp', sFilesRest, [], ...
        'timewindow',  [], ...
        'eventname',   '40-240Hz', ...
        'eventtime',   [], ...
        'bandpass',    [40, 240], ...
        'sensortypes', 'MEG', ...
        'usessp',      1, ...
        'saveerp',     0, ...
        'method',      1, ...  % PCA: One component per sensor
        'select',      1);

    % Process: Snapshot: Sensors/MRI registration
    bst_process('CallProcess', 'process_snapshot', sFilesRest, [], ...
        'target',         1, ...  % Sensors/MRI registration
        'modality',       1, ...  % MEG (All)
        'orient',         1);  % left

    % Process: Snapshot: SSP projectors
    bst_process('CallProcess', 'process_snapshot', sFilesRest, [], ...
        'target',         2, ...  % SSP projectors
        'modality',       1);     % MEG (All)

    % ===== SOURCE ESTIMATION =====
    % Process: Select file names with tag: task-rest
    sFilesNoise = bst_process('CallProcess', 'process_select_tag', sFilesBand, [], ...
        'tag',    '1-Rnoise', ...
        'search', 1, ...  % Search the file names
        'select', 1);  % Select only the files with the tag


    sFilesNoiseFull = bst_process('CallProcess', 'process_import_data_time', sFilesNoise, [], ...
        'subjectname',  SubjectNames{iSubj}, ...
        'condition',    '', ...
        'datafile',     {'', ''}, ...
        'timewindow',   [], ...
        'split',        0, ...
        'ignoreshort',  0, ...
        'channelalign', 0, ...
        'usectfcomp',   0, ...
        'usessp',       0, ...
        'freq',         [], ...
        'baseline',     []);

    sFilesRestFull = bst_process('CallProcess', 'process_import_data_time', sFilesRest, [], ...
        'subjectname',  SubjectNames{iSubj}, ...
        'condition',    '', ...
        'datafile',     {'', ''}, ...
        'timewindow',   [], ...
        'split',        0, ...
        'ignoreshort',  0, ...
        'channelalign', 0, ...
        'usectfcomp',   0, ...
        'usessp',       0, ...
        'freq',         [], ...
        'baseline',     []);

    % Process: Compute covariance (noise or data)
    bst_process('CallProcess', 'process_noisecov',  sFilesNoiseFull, [], ...
        'baseline',       [], ...
        'sensortypes',    'MEG', ...
        'target',         1, ...  % Noise covariance     (covariance over baseline time window)
        'dcoffset',       1, ...  % Block by block, to avoid effects of slow shifts in data
        'identity',       0, ...
        'copycond',       1, ...
        'copysubj',       0, ...
        'replacefile',    1);  % Replace
    
    % get noise covariance matrix for SNR estimation
    path_brainstorm_db = '/path/to/brainstorm_db/';
    noiseCovPath = fullfile(path_brainstorm_db, 'megHCP/data/', ...
        SubjectNames{iSubj}, '/1-Rnoise_c_rfDC_resample_notch_high/noisecov_full.mat');
    noiseCovData = load(noiseCovPath);
    noiseCovMat = noiseCovData.NoiseCov;

    temp = sFilesRest.FileName;
    sTime = load(file_fullpath(temp), 'Time');
    bst_process('CallProcess', 'process_noisecov', sFilesRestFull, [], ...
        'baseline',       [sTime.Time(1) sTime.Time(end)], ...
        'datatimewindow', [sTime.Time(1) sTime.Time(end)], ...
        'sensortypes',    'MEG', ...
        'target',         2, ...  % Data covariance      (covariance over data time window)
        'dcoffset',       1, ...  % Block by block, to avoid effects of slow shifts in data
        'identity',       0, ...
        'copycond',       1, ...
        'copysubj',       0, ...
        'copymatch',      0, ...
        'replacefile',    1);  % Replace

    % Process: Compute head model
    bst_process('CallProcess', 'process_headmodel', sFilesRestFull, [], ...
        'sourcespace', 1, ...  % Cortex surface
        'meg',         3);     % Overlapping spheres

    % Process: Compute sources [2018]
    sSrcRest = bst_process('CallProcess', 'process_inverse_2018', sFilesRestFull, [], ...
        'output',  3, ...  % Kernel only: one per file: 2; Full results: 3
        'inverse', struct(...
             'Comment',        'PNAI: MEG', ...
             'InverseMethod',  'lcmv', ...
             'InverseMeasure', 'nai', ...
             'SourceOrient',   {{'fixed'}}, ...
             'Loose',          0.2, ...
             'UseDepth',       1, ...
             'WeightExp',      0.5, ...
             'WeightLimit',    10, ...
             'NoiseMethod',    'median', ...
             'NoiseReg',       0.1, ...
             'SnrMethod',      'rms', ...
             'SnrRms',         1e-06, ...
             'SnrFixed',       3, ...
             'ComputeKernel',  0, ... % change to 1 for Kernel only and to 0 for Full results
             'DataTypes',      {{'MEG'}}));
         
    % Save Vertex x Time
    tsMatrix = file_fullpath(sSrcRest.FileName);
    outpath = fullfile(hcp_dir, 'brainstormResults', ...
        'vertexTimeSeries', SubjectNames{iSubj});
    if ~isfolder(outpath)
        mkdir(outpath)
%         copyfile(tsMatrix, fullfile(outpath, strcat(SubjectNames{iSubj}, ...
%             '_vertexTS.mat')))
        movefile(tsMatrix, fullfile(outpath, strcat(SubjectNames{iSubj}, ...
            '_vertexTS.mat')))
    else
%         copyfile(tsMatrix, fullfile(outpath, strcat(SubjectNames{iSubj}, ...
%             '_vertexTS.mat')))
        movefile(tsMatrix, fullfile(outpath, strcat(SubjectNames{iSubj}, ...
            '_vertexTS.mat')))
    end
    
    
    % get model-based SNR estimation
     sourcedata = in_bst_results(sSrcRest.FileName);
     
     sStudy = bst_get('Study');
     headmodel = sStudy.HeadModel.FileName;
     sHeadModel = in_bst_headmodel(headmodel);
     Gain_constrained = bst_gain_orient(sHeadModel.Gain, sHeadModel.GridOrient);
     gain = Gain_constrained(sourcedata.GoodChannel, :);
     
     n_channels = size(noiseCovMat, 1);  % number of sensors/channels
     b_k2 = (gain .* gain)';
     s_k2 = diag(noiseCovMat);
     s_k2 = s_k2(sourcedata.GoodChannel, :);
     scaling = (1 / n_channels) .* sum(b_k2 ./ s_k2', 2); % scaling is changed during parcellation in snr_parcellation.py
     snr = 10 .* log10((10 * 10) .* scaling);
          

     % Save Vertex SNR
     outpath = fullfile(hcp_dir, 'brainstormResults', ...
         'snrData_lcmv', SubjectNames{iSubj});
     if ~isfolder(outpath)
         mkdir(outpath)
         save(fullfile(outpath, strcat(SubjectNames{iSubj}, ...
             '_snrData.mat')), 'snr')
     else
         save(fullfile(outpath, strcat(SubjectNames{iSubj}, ...
             '_snrData.mat')), 'snr')
     end

    % ===== POWER MAPS =====
    % Process: Power spectrum density (Welch) (this saves out full power
    % spectrum without taking the band-limited average; see below for
    % band-limired power)
    sSrcPsd = bst_process('CallProcess', 'process_psd', sSrcRest, [], ...
        'timewindow',  [sTime.Time(1) sTime.Time(end)], ...
        'win_length',  4, ...
        'win_overlap', 50, ...
        'units',       'physical', ...  % Physical: U2/Hz
        'clusters',    {}, ...
        'scoutfunc',   1, ...  % Mean
        'edit',        struct(...
             'Comment',         'Power', ...   % 'Power,FreqBands' for avg power in freq bands; 'Power' for full freq power
             'TimeBands',       [], ...
             'Freqs',           [], ...
             'ClusterFuncTime', 'none', ...
             'Measure',         'power', ...
             'Output',          'all', ...
             'SaveKernel',      0));

    % Process: Spectrum normalization
    sSrcPsdNorm = bst_process('CallProcess', 'process_tf_norm', sSrcPsd, [], ...
        'normalize', 'relative', ...  % Relative power (divide by total power)
        'overwrite', 0);

    % Process: Spatial smoothing (3.00)
    sSrcPsdNormSmooth = bst_process('CallProcess', 'process_ssmooth_surfstat', sSrcPsdNorm, [], ...
        'fwhm',      3, ...
        'overwrite', 0);

    % Save Vertex x Normalized Power
    psdMatrixSmooth = file_fullpath(sSrcPsdNormSmooth.FileName);
    outpath = fullfile(hcp_dir, 'brainstormResults', ...
        'sourcePsdNormFull4sec_resamp', SubjectNames{iSubj});
    if ~isfolder(outpath)
        mkdir(outpath)
        copyfile(psdMatrixSmooth, fullfile(outpath, strcat(SubjectNames{iSubj}, ...
            '_psd_norm_smooth3mm.mat')))
    else
        copyfile(psdMatrixSmooth, fullfile(outpath, strcat(SubjectNames{iSubj}, ...
            '_psd_norm_smooth3mm.mat')))
    end

    
    % Process: Power spectrum density (Welch) 
    % (this saves out band-limited power maps)
    sSrcPsd_band = bst_process('CallProcess', 'process_psd', sSrcRest, [], ...
        'timewindow',  [sTime.Time(1) sTime.Time(end)], ...
        'win_length',  4, ...
        'win_overlap', 50, ...
        'units',       'physical', ...  % Physical: U2/Hz
        'clusters',    {}, ...
        'scoutfunc',   1, ...  % Mean
        'edit',        struct(...
             'Comment',         'Power,FreqBands', ...   % 'Power,FreqBands' for avg power in freq bands; 'Power' for full freq power
             'TimeBands',       [], ...
             'Freqs',           {{'delta', '2, 4', 'mean'; ...
                                  'theta', '5, 7', 'mean'; ...
                                  'alpha', '8, 12', 'mean'; ...
                                  'beta', '15, 29', 'mean'; ...
                                  'gamma1', '30, 59', 'mean'; ...
                                  'gamma2', '60, 90', 'mean'}}, ...
             'ClusterFuncTime', 'none', ...
             'Measure',         'power', ...
             'Output',          'all', ...
             'SaveKernel',      0));

    % Process: Spectrum normalization
    sSrcPsdNorm_band = bst_process('CallProcess', 'process_tf_norm', sSrcPsd_band, [], ...
        'normalize', 'relative', ...  % Relative power (divide by total power)
        'overwrite', 0);

    % Process: Spatial smoothing (3.00)
    sSrcPsdNormSmooth_band = bst_process('CallProcess', 'process_ssmooth_surfstat', sSrcPsdNorm_band, [], ...
        'fwhm',      3, ...
        'overwrite', 0);

    % Save Vertex x Normalized Power
    psdMatrixSmooth = file_fullpath(sSrcPsdNormSmooth_band.FileName);
    outpath = fullfile(hcp_dir, 'brainstormResults', ...
        'sourcePsdNormBand4sec_resamp', SubjectNames{iSubj});
    if ~isfolder(outpath)
        mkdir(outpath)
        copyfile(psdMatrixSmooth, fullfile(outpath, strcat(SubjectNames{iSubj}, ...
            '_psd_norm_smooth3mm.mat')))
    else
        copyfile(psdMatrixSmooth, fullfile(outpath, strcat(SubjectNames{iSubj}, ...
            '_psd_norm_smooth3mm.mat')))
    end
    
    
    % Save report
    ReportFile = bst_report('Save', []);
    if ~isempty(reports_dir) && ~isempty(ReportFile)
        bst_report('Export', ReportFile, bst_fullfile(reports_dir, ...
            ['report_' ProtocolName '_' SubjectNames{iSubj} '.html']));
    end
    
    toc

end
