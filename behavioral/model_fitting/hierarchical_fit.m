fname = 'task_data_indiv_aff_score_precise_rt_action_manual.json';
str = fileread(fname);
data_json = jsondecode(str);
n_subj = size(data_json);
n_subj = n_subj(1);
data = cell(n_subj,1);
for i = 1:n_subj
    data{i} = data_json(i);
end

% and so on
addpath(fullfile('models'));
addpath('~/Documents/MATLAB/cbm/codes');

models = {
    @RL,
    @RL_bias, 
    @RL_prior,
    @RL_cost,
    @RL_fixed,
    @RL_conflict,
    @RL_reliability,
    @RL_performance
    };
fcbm_maps = {
    'lap_out/lap_RL_indiv_aff_score_precise_rt_action_manual.mat',
    'lap_out/lap_RL_bias_indiv_aff_score_precise_rt_action_manual.mat', 
    'lap_out/lap_RL_prior_indiv_aff_score_precise_rt_action_manual.mat',
    'lap_out/lap_RL_cost_indiv_aff_score_precise_rt_action_manual.mat',
    'lap_out/lap_RL_fixed_indiv_aff_score_precise_rt_action_manual.mat',
    'lap_out/lap_RL_conflict_indiv_aff_score_precise_rt_action_manual.mat',
    'lap_out/lap_RL_reliability_indiv_aff_score_precise_rt_action_manual.mat',
    'lap_out/lap_RL_performance_indiv_aff_score_precise_rt_action_manual.mat',
    };

fname_hbi = 'hbi_out/hbi_RL_indiv_aff_score_precise_rt_action_manual.mat';
cbm_hbi(data, models, fcbm_maps, fname_hbi);
cbm_hbi_null(data,fname_hbi);
