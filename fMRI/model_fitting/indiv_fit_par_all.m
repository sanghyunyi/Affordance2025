delete(gcp('nocreate'));
fname = 'task_data_avg_aff_score_precise_rt_all.json';
str = fileread(fname);
data_json = jsondecode(str);
n_subj = size(data_json);
n_subj = n_subj(1);

data = cell(n_subj,1);
for i = 1:n_subj
    data{i} = data_json(i);
end
    
% Add required paths
addpath(fullfile('models'));
addpath('~/Documents/MATLAB/cbm/codes');

% Create a parallel pool
parpool(12);

% Define models to run and their parameters
models = {
    struct('name', 'RL', 'n_par', 5);
    struct('name', 'RL_bias', 'n_par', 6);
    struct('name', 'RL_prior', 'n_par', 6);
    struct('name', 'RL_reliability', 'n_par', 10);
    struct('name', 'RL_conflict', 'n_par', 8);
    struct('name', 'RL_fixed', 'n_par', 7);
    struct('name', 'RL_performance', 'n_par', 8);
    struct('name', 'RL_cost', 'n_par', 8);
    struct('name', 'Bayesian_aff_as_bias', 'n_par', 9);
};

results = cell(length(models), 3); % Store model name, loglik, and log_evidence

% Process each model
for m = 1:length(models)
    model = models{m};
    model_name = model.name;
    n_par = model.n_par;
    
    % Run individual fits
    parfor i = 1:n_subj
        data_subj = data(i);
        prior_RL = struct('mean', zeros(n_par, 1), 'variance', 6.25);
        fname_RL = fullfile('lap_subjects', append('lap_', model_name, '_', num2str(i), '.mat'));
        cbm_lap(data_subj, str2func(model_name), prior_RL, fname_RL);
    end

    % Aggregate results
    fname_subjs = cell(n_subj,1);
    for n = 1:n_subj
        fname_subjs{n} = fullfile('lap_subjects', ['lap_', model_name, '_', num2str(n), '.mat']);
    end
    
    fname_RL = fullfile('lap_out', append('lap_', model_name, '_avg_aff_score_precise_rt.mat'));
    cbm_lap_aggregate(fname_subjs, fname_RL);

    % Load and compute log-likelihood and log-evidence
    fname = load(fname_RL);
    cbm = fname.cbm;
    loglik = sum(cbm.output.loglik);
    log_evidence = sum(cbm.output.log_evidence);
    
    % Store results
    results{m, 1} = model_name;
    results{m, 2} = loglik;
    results{m, 3} = log_evidence;
end

% Print all results together
fprintf('\nModel Results:\n');
for m = 1:length(models)
    fprintf('Model: %s\n', results{m, 1});
    fprintf('Total log-likelihood: %f\n', results{m, 2});
    fprintf('Total log-evidence: %f\n\n', results{m, 3});
end
