delete(gcp('nocreate'));
load 'sim_data.mat'

% and so on
addpath(fullfile('models'));
addpath('~/Documents/MATLAB/cbm/codes');

n_sim = 100;

n_model = 7;
bic = cell(n_model, n_model);

parpool(4);

for m = 1:n_model
    data = sim_data{m};
    %-----------------------------

    parfor i = 1:n_sim
        data_subj = data(i);
        prior_RL = struct('mean',zeros(5,1),'variance', 6.25); % note dimension of 'mean' 
        fname_RL = fullfile('lap_subjects', append('lap_RL_', num2str(i), '.mat'));
        cbm_lap(data_subj, @RL, prior_RL, fname_RL); % specify model here
    end

    fname_subjs = cell(n_sim,1);
    for n=1:length(fname_subjs)
        fname_subjs{n} = fullfile('lap_subjects',['lap_RL_' num2str(n) '.mat']);
    end

    fname_RL = 'lap_out/lap_RL_aff_model_recov.mat'; 
    cbm_lap_aggregate(fname_subjs, fname_RL);

    fname = load(fname_RL);
    cbm = fname.cbm;
    %bic{m, 1} = -2*cbm.output.loglik+5*log(20*48);
    bic{m, 1} = -2*cbm.output.log_evidence;
    %-----------------------------

    %parpool(4);

    parfor i = 1:n_sim
        data_subj = data(i);
        prior_RL = struct('mean',zeros(6,1),'variance', 6.25); % note dimension of 'mean' 
        fname_RL = fullfile('lap_subjects', append('lap_RL_', num2str(i), '.mat'));
        cbm_lap(data_subj, @RL_bias, prior_RL, fname_RL); % specify model here
    end

    fname_subjs = cell(n_sim,1);
    for n=1:length(fname_subjs)
        fname_subjs{n} = fullfile('lap_subjects',['lap_RL_' num2str(n) '.mat']);
    end

    fname_RL = 'lap_out/lap_RL_aff_model_recov.mat'; 
    cbm_lap_aggregate(fname_subjs, fname_RL);

    fname = load(fname_RL);
    cbm = fname.cbm;
    %bic{m, 2} = -2*cbm.output.loglik+6*log(20*48);
    bic{m, 2} = -2*cbm.output.log_evidence;
    %-----------------------------
    %parpool(4);

    parfor i = 1:n_sim
        data_subj = data(i);
        prior_RL = struct('mean',zeros(6,1),'variance', 6.25); % note dimension of 'mean' 
        fname_RL = fullfile('lap_subjects', append('lap_RL_', num2str(i), '.mat'));
        cbm_lap(data_subj, @RL_prior, prior_RL, fname_RL); % specify model here
    end

    fname_subjs = cell(n_sim,1);
    for n=1:length(fname_subjs)
        fname_subjs{n} = fullfile('lap_subjects',['lap_RL_' num2str(n) '.mat']);
    end

    fname_RL = 'lap_out/lap_RL_aff_model_recov.mat'; 
    cbm_lap_aggregate(fname_subjs, fname_RL);

    fname = load(fname_RL);
    cbm = fname.cbm;
    %bic{m, 3} = -2*cbm.output.loglik+6*log(20*48);
    bic{m, 3} = -2*cbm.output.log_evidence;
    %-----------------------------
    %parpool(4);

    parfor i = 1:n_sim
        data_subj = data(i);
        prior_RL = struct('mean',zeros(8,1),'variance', 6.25); % note dimension of 'mean' 
        fname_RL = fullfile('lap_subjects', append('lap_RL_', num2str(i), '.mat'));
        cbm_lap(data_subj, @RL_conflict, prior_RL, fname_RL); % specify model here
    end

    fname_subjs = cell(n_sim,1);
    for n=1:length(fname_subjs)
        fname_subjs{n} = fullfile('lap_subjects',['lap_RL_' num2str(n) '.mat']);
    end

    fname_RL = 'lap_out/lap_RL_aff_model_recov.mat'; 
    cbm_lap_aggregate(fname_subjs, fname_RL);

    fname = load(fname_RL);
    cbm = fname.cbm;
    %bic{m, 4} = -2*cbm.output.loglik+9*log(20*48);
    bic{m, 4} = -2*cbm.output.log_evidence;
    %-----------------------------

    %parpool(4);

    parfor i = 1:n_sim
        data_subj = data(i);
        prior_RL = struct('mean',zeros(10,1),'variance', 6.25); % note dimension of 'mean' 
        fname_RL = fullfile('lap_subjects', append('lap_RL_', num2str(i), '.mat'));
        cbm_lap(data_subj, @RL_reliability, prior_RL, fname_RL); % specify model here
    end

    fname_subjs = cell(n_sim,1);
    for n=1:length(fname_subjs)
        fname_subjs{n} = fullfile('lap_subjects',['lap_RL_' num2str(n) '.mat']);
    end

    fname_RL = 'lap_out/lap_RL_aff_model_recov.mat'; 
    cbm_lap_aggregate(fname_subjs, fname_RL);

    fname = load(fname_RL);
    cbm = fname.cbm;
    %bic{m, 5} = -2*cbm.output.loglik+10*log(20*48);
    bic{m, 5} = -2*cbm.output.log_evidence;
    %-----------------------------

    %parpool(4);

    parfor i = 1:n_sim
        data_subj = data(i);
        prior_RL = struct('mean',zeros(7,1),'variance', 6.25); % note dimension of 'mean' 
        fname_RL = fullfile('lap_subjects', append('lap_RL_', num2str(i), '.mat'));
        cbm_lap(data_subj, @RL_fixed, prior_RL, fname_RL); % specify model here
    end

    fname_subjs = cell(n_sim,1);
    for n=1:length(fname_subjs)
        fname_subjs{n} = fullfile('lap_subjects',['lap_RL_' num2str(n) '.mat']);
    end

    fname_RL = 'lap_out/lap_RL_aff_model_recov.mat'; 
    cbm_lap_aggregate(fname_subjs, fname_RL);

    fname = load(fname_RL);
    cbm = fname.cbm;
    %bic{m, 6} = -2*cbm.output.loglik+7*log(20*48);
    bic{m, 6} = -2*cbm.output.log_evidence;
    %-----------------------------

    %parpool(4);

    parfor i = 1:n_sim
        data_subj = data(i);
        prior_RL = struct('mean',zeros(8,1),'variance', 6.25); % note dimension of 'mean' 
        fname_RL = fullfile('lap_subjects', append('lap_RL_', num2str(i), '.mat'));
        cbm_lap(data_subj, @RL_performance, prior_RL, fname_RL); % specify model here
    end

    fname_subjs = cell(n_sim,1);
    for n=1:length(fname_subjs)
        fname_subjs{n} = fullfile('lap_subjects',['lap_RL_' num2str(n) '.mat']);
    end

    fname_RL = 'lap_out/lap_RL_aff_model_recov.mat'; 
    cbm_lap_aggregate(fname_subjs, fname_RL);

    fname = load(fname_RL);
    cbm = fname.cbm;
    %bic{m, 6} = -2*cbm.output.loglik+7*log(20*48);
    bic{m, 7} = -2*cbm.output.log_evidence;
    %-----------------------------

end

save model_recov_results/log_evidence.mat bic
