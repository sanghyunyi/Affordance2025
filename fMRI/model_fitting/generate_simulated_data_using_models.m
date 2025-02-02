delete(gcp('nocreate'));
fname = 'task_data_sim.json';
str = fileread(fname);
data_json = jsondecode(str);
n_subj = size(data_json);
n_subj = n_subj(1);
n_sim = 100;

n_model = 7;
sim_data = cell(n_model, 1);

% and so on
addpath(fullfile('models'));
addpath('~/Documents/MATLAB/cbm/codes');

%-----------------------------

data = cell(n_subj, 1);
for i = 1:n_subj
    data{i} = data_json(i);
end

load 'lap_out/lap_RL_indiv_aff_score_precise_rt_action_manual.mat';
fit_params = cbm.output.parameters;
m = mean(fit_params);
s = std(fit_params);

output = cell(n_sim, 1);
target_params = zeros(n_sim, 5);
for i = 1:n_sim
    % data for each subject 
    subj = data{rem(i,n_subj)+1};
    parameters = m+randn(1,5).*s;
    target_params(i, :) = parameters;
    subj = gen_var_RL_sim(parameters, subj); % Target model
    output{i} = subj;
end

sim_data{1} = output;

%-----------------------------

data = cell(n_subj, 1);
for i = 1:n_subj
    data{i} = data_json(i);
end

load 'lap_out/lap_RL_bias_indiv_aff_score_precise_rt_action_manual.mat';
fit_params = cbm.output.parameters;
m = mean(fit_params);
s = std(fit_params);

output = cell(n_sim, 1);
target_params = zeros(n_sim, 6);
for i = 1:n_sim
    % data for each subject 
    subj = data{rem(i,n_subj)+1};
    parameters = m+randn(1,6).*s;
    target_params(i, :) = parameters;
    subj = gen_var_RL_bias_sim(parameters, subj); % Target model
    output{i} = subj;
end

sim_data{2} = output;

%-----------------------------

data = cell(n_subj, 1);
for i = 1:n_subj
    data{i} = data_json(i);
end

load 'lap_out/lap_RL_prior_indiv_aff_score_precise_rt_action_manual.mat';
fit_params = cbm.output.parameters;
m = mean(fit_params);
s = std(fit_params);

output = cell(n_sim, 1);
target_params = zeros(n_sim, 6);
for i = 1:n_sim
    % data for each subject 
    subj = data{rem(i,n_subj)+1};
    parameters = m+randn(1,6).*s;
    target_params(i, :) = parameters;
    subj = gen_var_RL_prior_sim(parameters, subj); % Target model
    output{i} = subj;
end

sim_data{3} = output;

%-----------------------------

data = cell(n_subj, 1);
for i = 1:n_subj
    data{i} = data_json(i);
end

load 'lap_out/lap_RL_conflict_indiv_aff_score_precise_rt_action_manual.mat';
fit_params = cbm.output.parameters;
m = mean(fit_params);
s = std(fit_params);

output = cell(n_sim, 1);
target_params = zeros(n_sim, 8);
for i = 1:n_sim
    % data for each subject 
    subj = data{rem(i,n_subj)+1};
    parameters = m+randn(1,8).*s;    
    target_params(i, :) = parameters;
    subj = gen_var_RL_conflict_sim(parameters, subj); % Target model
    output{i} = subj;
end

sim_data{4} = output;

%-----------------------------


data = cell(n_subj, 1);
for i = 1:n_subj
    data{i} = data_json(i);
end

load 'lap_out/lap_RL_reliability_indiv_aff_score_precise_rt_action_manual.mat';
fit_params = cbm.output.parameters;
m = mean(fit_params);
s = std(fit_params);

output = cell(n_sim, 1);
target_params = zeros(n_sim, 10);
for i = 1:n_sim
    % data for each subject 
    subj = data{rem(i,n_subj)+1};
    parameters = m+randn(1,10).*s;
    target_params(i, :) = parameters;
    subj = gen_var_RL_reliability_sim(parameters, subj); % Target model
    output{i} = subj;
end

sim_data{5} = output;

%-----------------------------


data = cell(n_subj, 1);
for i = 1:n_subj
    data{i} = data_json(i);
end

load 'lap_out/lap_RL_fixed_indiv_aff_score_precise_rt_action_manual.mat';
fit_params = cbm.output.parameters;
m = mean(fit_params);
s = std(fit_params);

output = cell(n_sim, 1);
target_params = zeros(n_sim, 7);
for i = 1:n_sim
    % data for each subject 
    subj = data{rem(i,n_subj)+1};
    parameters = m+randn(1,7).*s;
    target_params(i, :) = parameters;
    subj = gen_var_RL_fixed_sim(parameters, subj); % Target model
    output{i} = subj;
end

sim_data{6} = output;

%-----------------------------


data = cell(n_subj, 1);
for i = 1:n_subj
    data{i} = data_json(i);
end

load 'lap_out/lap_RL_performance_indiv_aff_score_precise_rt_action_manual.mat';
fit_params = cbm.output.parameters;
m = mean(fit_params);
s = std(fit_params);

output = cell(n_sim, 1);
target_params = zeros(n_sim, 8);
for i = 1:n_sim
    % data for each subject 
    subj = data{rem(i,n_subj)+1};
    parameters = m+randn(1,8).*s;
    target_params(i, :) = parameters;
    subj = gen_var_RL_performance_sim(parameters, subj); % Target model
    output{i} = subj;
end

sim_data{7} = output;

%-----------------------------

save sim_data.mat, sim_data
