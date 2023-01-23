%% process HCP data with Brainstorm
hcp_dir = '/path/to/megdata/and/results/';
reports_dir = strcat(hcp_dir, 'brainstormReports/');

loadedsubj = load(fullfile(hcp_dir, 'myMEGList.mat'));
subjList = split(loadedsubj.myMEG, '_');
subjList = subjList(:,2);

channels = load(fullfile(hcp_dir, 'myMEGbadChannels'));
badChannels = channels.BadChannels;

addpath(genpath('/usr/local/brainstorm3/'));

% Run Brainstorm
fcn_hcp_meg_process_rest(hcp_dir, subjList, badChannels, reports_dir) 

%% hctsa on MEG: parcellate
% requires cifti-matlab
% (https://github.com/Washington-University/cifti-matlab)
% addpath(genpath('/home/gshafiei/data1/Projects/packages/cifti-matlab'));
hcp_dir = '/path/to/megdata/and/results/';
datapath = strcat(hcp_dir, 'brainstormResults/vertexTimeSeries');
tspath = strcat(hcp_dir, 'parcellated/HCP_MEG_TimeSeries/Schaefer100/');
parcelpath = '/path/to/SchaeferParcellation/';

loadedsubj = load(fullfile(hcp_dir, 'myMEGList.mat'));
subjList = split(loadedsubj.myMEG, '_');
subjList = subjList(:,2);

% 100 parcels
labels = num2str([1:100]');
labels = cellstr(labels);
keywords = strcat(num2str([1:100]'),',Schefer100,meg');
keywords = cellstr(keywords);

% load parcellation
parcellationL = gifti(strcat(parcelpath,'fslr4k/Schaefer100_L.4k.label.gii'));
parcellationR = gifti(strcat(parcelpath,'fslr4k/Schaefer100_R.4k.label.gii'));

p = [parcellationL.cdata; parcellationR.cdata];
parcelIDs = unique(p); parcelIDs(parcelIDs<1) = [];


% load subject time series
for iSubj = 1:length(subjList)
    tic
    subjTS = load(fullfile(datapath, strcat(subjList{iSubj}, '/', ...
        subjList{iSubj}, '_vertexTS.mat')));
    
    % parcellate using PCA
    data = subjTS.ImageGridAmp;
    parcellatedData = zeros(length(parcelIDs), size(data, 2));
    for IDnum = 1:length(parcelIDs)
        [pcaTS, ~] = PcaFirstMode_fromBST(data(p==parcelIDs(IDnum), :));
        parcellatedData(IDnum, :) = pcaTS;
    end
    
    % z-score time series for each node
    timeSeriesData = zscore(parcellatedData')';

    
    % save for hctsa
    save(fullfile(tspath,[subjList{iSubj},'_meg_pca_Schaefer100.mat']),...
        'keywords', 'labels', 'timeSeriesData');
    clear timeSeriesData    
    
    % delete original file
    delete(fullfile(datapath, strcat(subjList{iSubj}, '/', ...
        subjList{iSubj}, '_vertexTS.mat')));
    fprintf('\nSubj%i - done!\n',iSubj)
    toc
end

%% run hctsa on data segments
hcp_dir = '/path/to/megdata/and/results/';
tspath = strcat(hcp_dir, 'parcellated/HCP_MEG_TimeSeries/Schaefer100/');
outpath = strcat(hcp_dir, 'HCP_MEG_outputs/Schaefer100/');

segTSpath = strcat(tspath, 'dataSegments/');
segOutpath = strcat(outpath, 'dataSegments/');

loadedsubj = load(fullfile(hcp_dir, 'myMEGList.mat'));
subjList = split(loadedsubj.myMEG, '_');
subjList = subjList(:,2);


% first load data for one subject
for iSubj = 1:length(subjList)
    subjFile = strcat(tspath,[subjList{iSubj},'_meg_pca_Schaefer100.mat']);
    subjData = load(subjFile);
    keywords = subjData.keywords;
    labels = subjData.labels;

    % new sampling rate 508.6275 Hz
    % how many samples is 5 seconds:
    stepLength = floor(5 * 508.6275);

    % let's drop first 30 seconds or 6 steps
    subjData.timeSeriesData(:, 1:6*stepLength) = [];

    % save new data in increments of 5 seconds, from 5 to 75 seconds
    for i = 1:25
        timeSeriesData = subjData.timeSeriesData(:, 1:i*stepLength);
        % z-score time series for each node
        timeSeriesData = zscore(timeSeriesData')';

        % save for hctsa
        temp = strcat(subjList{iSubj},'_meg_pca_Schaefer100_seg',num2str(i,'%02.f'),'.mat');
        save(fullfile(segTSpath,temp),...
            'keywords', 'labels', 'timeSeriesData');
    end
end

% run hctsa on data segments (takes time!!)
% get filenames; here only for 80seconds segment (i.e. seg16)
files = dir(strcat(segTSpath, '*_seg16.mat'));
files = struct2cell(files);
fileNames = files(1,:);

parfor (file = 1:length(fileNames),33)
    tic
    inputFile = strcat(segTSpath,fileNames{file});
    outputFile = strcat(segOutpath, 'fullSet_80/', fileNames{file});
    temp = split(outputFile, '.');
    if ~isfile(strcat(temp(1), '_N.', temp(2)))
        TS_Init(inputFile,'./Database/INP_mops_modified.txt','./Database/INP_ops_modified.txt',0,outputFile);
        TS_Compute(0,[],[],'missing',outputFile,0);
        TS_Normalize([],[],outputFile,[]);
    end
    fprintf('\nFile%i - done!\n',file)
    toc
end

%% table2struct conversion for python analysis
hcp_dir = '/path/to/megdata/and/results/';
datapath = strcat(hcp_dir, 'HCP_MEG_outputs/Schaefer100/');

segOutpath = strcat(datapath, 'dataSegments/fullSet_80/');

% get subj IDs
files = dir(strcat(segOutpath, '*_N.mat'));
files = struct2cell(files);
fileNames = files(1,:);

for file = 1:length(fileNames)
    tic
    inputFile = strcat(segOutpath,fileNames{file});
    outputFile = strcat(segOutpath,fileNames{file});
    
    temp = load(inputFile);
    
    temp.Operations = table2struct(temp.Operations);
    temp.MasterOperations = table2struct(temp.MasterOperations);

    save(outputFile, '-v7.3', '-struct', 'temp')
    
    fprintf('\nFile%i - done!\n',file)
    toc
end
