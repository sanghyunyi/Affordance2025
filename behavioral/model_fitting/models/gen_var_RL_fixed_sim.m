function [subj] = model_RL(parameters,subj)
% lr for slot and cue
nd_alpha  = parameters(1); % normally-distributed alpha
alpha     = 1/(1+exp(-nd_alpha)); % alpha (transformed to be between zero and one)

% inverse temp for cue
nd_beta = parameters(2);
beta    = exp(nd_beta);

nd_beta2 = parameters(3);
beta2    = exp(nd_beta2);

nd_beta3 = parameters(4);
beta3    = nd_beta3;

nd_beta4 = parameters(5);
beta4    = nd_beta4;

nd_beta5 = parameters(6);
beta5    = nd_beta5;

nd_w_c  = parameters(7); % normally-distributed alpha
w_c     = 1/(1+exp(-nd_w_c)); % alpha (transformed to be between zero and one)

% unpack data
%actions = subj.response_list; % 1 for action=f and 2 for action=j # from subj
%outcome = subj.reward_list; % 1 for outcome=1 and 0 for outcome=0 # from subj
%actions = subj.response_list; % 1 for action=f and 2 for action=j
%outcome = subj.reward_list; % 1 for outcome=1 and 0 for outcome=0
out = subj.out_list; % 1 for outcome=1 and 0 for outcome=0 # from task design
aff = subj.affordance_list; % pinch, clench, poke, palm, familiarity scores
states = subj.state_list; % stim idxs 
is_new_block = subj.is_new_block;

%===============================================================================
%===============================================================================

% number of trials
T       = size(out,1);

% Q-value for each stim
q       = zeros(16*3, 3); % Q-value initialized at 0


% to save probability of choice. Currently NaNs, will be filled below
p       = nan(T,1);
outcome = nan(T,1);
actions = nan(T,1);

chosen_aff = nan(T, 1);
chosen_q = nan(T, 1);

all_aff = nan(T, 6);
all_q = nan(T, 6);

wc_list = nan(T,1);
rpe_aff_list = nan(T,1);
rpe_q_list = nan(T,1);


for t=1:T
    state = states(t);
    q_pinch = q(state, 1);
    aff_pinch = aff(t, 1)/(100*beta2);
    q_clench = q(state, 2);
    aff_clench = aff(t, 2)/(100*beta2);
    q_poke = q(state, 3);
    aff_poke = aff(t, 3)/(100*beta2);

    p_pinch = exp(beta*q_pinch+beta3)/(exp(beta*q_pinch+beta3)+exp(beta*q_clench+beta4)+exp(beta*q_poke+beta5));
    p_clench = exp(beta*q_clench+beta4)/(exp(beta*q_pinch+beta3)+exp(beta*q_clench+beta4)+exp(beta*q_poke+beta5));
    p_poke = 1 - p_pinch - p_clench;

    p_pinch_aff = exp(beta*aff_pinch+beta3)/(exp(beta*aff_pinch+beta3)+exp(beta*aff_clench+beta4)+exp(beta*aff_poke+beta5));
    p_clench_aff = exp(beta*aff_clench+beta4)/(exp(beta*aff_pinch+beta3)+exp(beta*aff_clench+beta4)+exp(beta*aff_poke+beta5));
    p_poke_aff = 1 - p_pinch_aff - p_clench_aff;

    %xi_d = abs(xi_q) - abs(xi_aff); % reliability difference, initialized at 0. xi_aff - xi_q

    all_aff(t, 1) = p_pinch_aff;
    all_aff(t, 4) = aff_pinch;
    all_q(t, 1) = p_pinch;
    all_q(t, 4) = q_pinch;
    all_aff(t, 2) = p_clench_aff;
    all_aff(t, 5) = aff_clench;
    all_q(t, 2) = p_clench;
    all_q(t, 5) = q_clench;
    all_aff(t, 3) = p_poke_aff;
    all_aff(t, 6) = aff_poke;
    all_q(t, 3) = p_poke;
    all_q(t, 6) = q_poke;

    p_pinch = p_pinch*w_c + p_pinch_aff*(1-w_c);
    p_clench = p_clench*w_c + p_clench_aff*(1-w_c);
    p_poke = p_poke*w_c + p_poke_aff*(1-w_c);
    

    r = rand(1);
    
    wc_list(t) = w_c;

    if r < p_pinch
        p(t) = p_pinch;
        actions(t) = 1;
        o = out(t, 1);

        chosen_q(t) = q_pinch;
        chosen_aff(t) = aff_pinch;

        delta = o - q_pinch;
        q(state, 1) = q(state, 1) + (alpha*delta);

        rpe_q = delta;
        rpe_aff = o - aff_pinch;
    elseif r < p_pinch + p_clench 
        p(t) = p_clench;
        actions(t) = 2;
        o = out(t, 2);

        chosen_q(t) = q_clench;
        chosen_aff(t) = aff_clench;

        delta = o - q_clench;
        q(state, 2) = q(state, 2) + (alpha*delta);

        rpe_q = delta;
        rpe_aff = o - aff_clench;
    elseif r >= p_pinch + p_clench
        p(t) = p_poke;
        actions(t) = 3;
        o = out(t, 3);

        chosen_q(t) = q_poke;
        chosen_aff(t) = aff_poke;

        delta = o - q_poke;
        q(state, 3) = q(state, 3) + (alpha*delta);

        rpe_q = delta;
        rpe_aff = o - aff_poke;
    else
        p(t) = 1/3;
        o = 0;
    end
    outcome(t) = o;
    rpe_q_list(t) = rpe_q;
    rpe_aff_list(t) = rpe_aff;
end

%===============================================================================
%===============================================================================

subj.reward_list = outcome;
subj.response_list = actions;
subj.gen_params = parameters;
subj.chosen_prob = p;
subj.all_q = all_q;
subj.chosen_q = chosen_q;
subj.all_aff = all_aff;
subj.chosen_aff = chosen_aff;
subj.wc_list = wc_list;
subj.rpe_aff_list = rpe_aff_list;
subj.rpe_q_list = rpe_q_list;
end
