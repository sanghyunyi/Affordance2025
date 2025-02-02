delete(gcp('nocreate'));
fname = 'task_data_avg_aff_score_precise_rt.json';
str = fileread(fname);
data_json = jsondecode(str);
n_subj = size(data_json);
n_subj = n_subj(1);
data = cell(n_subj,1);
for i = 1:n_subj
    data{i} = data_json(i);
end

load 'lap_out/lap_RL_performance_avg_aff_score_precise_rt.mat';

% and so on
addpath(fullfile('models'));
addpath('~/Documents/MATLAB/cbm/codes');

%------------------------------------

output = cell(n_subj, 1);
fit_params = cbm.output.parameters;
%fit_params = cbm.output.parameters{5};

for i = 1:n_subj
    % data for each subject 
    subj = data{i};
    parameters = fit_params(i, :); % this should be from the fit data
    subj2 = gen_var_RL_performance(parameters, subj); % Target model
    output{i} = subj2;
end

output2 = jsonencode(output);
fileID = fopen('./model_based_var/model_based_generated_var_RL_performance_avg_aff_score_precise_rt.json','w');
fprintf(fileID, output2);
fclose(fileID);
