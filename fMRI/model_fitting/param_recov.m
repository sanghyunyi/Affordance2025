delete(gcp('nocreate'));
%fname = 'task_data_sim.json';
fname = 'task_data_avg_aff_score_precise_rt.json';
str = fileread(fname);
data_json = jsondecode(str);
n_subj = size(data_json);
n_subj = n_subj(1);
n_sim = 1000;

n_param = 8; % Change it accordingly to each model
data = cell(n_sim,1);
for i = 1:n_subj
    data{i} = data_json(i);
end

% and so on
addpath(fullfile('models'));
addpath('~/Documents/MATLAB/cbm/codes');

load 'lap_out/lap_RL_performance_avg_aff_score_precise_rt.mat'; % Change it to the model you want to test

fit_params = cbm.output.parameters;
m = mean(fit_params)
s = std(fit_params)

target_params = zeros(n_sim, n_param);
for i = 1:n_sim
    % data for each subject 
    subj = data{rem(i,n_subj)+1};
    parameters = m+randn(1,n_param).*s;
    target_params(i, :) = parameters;
    subj = gen_var_RL_performance_sim(parameters, subj); % Change it to the model you want to test

    data{i} = subj;
end

parpool(4);

parfor i = 1:n_sim
    data_subj = data(i);
    prior_RL = struct('mean',zeros(n_param,1),'variance', 6.25); % note dimension of 'mean' 
    fname_RL = fullfile('lap_subjects', append('lap_RL_', num2str(i), '.mat'));
    cbm_lap(data_subj, @RL_performance, prior_RL, fname_RL); % Change it to the model you want to test
end

fname_subjs = cell(n_sim,1);
for n=1:length(fname_subjs)
    fname_subjs{n} = fullfile('lap_subjects',['lap_RL_' num2str(n) '.mat']);
end

fname_RL = 'param_recovery_plots/recov_lap_RL_performance_1000sim.mat'; 
cbm_lap_aggregate(fname_subjs, fname_RL);

fname = load(fname_RL);
cbm = fname.cbm;

recovered_params = cbm.output.parameters;

save param_recovery_plots/param_recov_performance_1000sim.mat target_params recovered_params

