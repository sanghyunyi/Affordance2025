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

% unpack data
%actions = subj.response_list; % 1 for action=f and 2 for action=j # from subj
%outcome = subj.reward_list; % 1 for outcome=1 and 0 for outcome=0 # from subj
out = subj.out_list; % 1 for outcome=1 and 0 for outcome=0
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

rpe_q_list = nan(T,1);

for t=1:T
    state = states(t);
    q_pinch = q(state, 1);
    aff_pinch = aff(t, 1)/(100*beta2);
    q_clench = q(state, 2);
    aff_clench = aff(t, 2)/(100*beta2);
    q_poke = q(state, 3);
    aff_poke = aff(t, 3)/(100*beta2);

    q_pinch_ = q_pinch + aff_pinch;
    q_clench_ = q_clench + aff_clench;
    q_poke_ = q_poke + aff_poke;

    p_pinch = exp(beta*q_pinch + beta3)/(exp(beta*q_pinch + beta3)+exp(beta*q_clench + beta4)+exp(beta*q_poke + beta5));
    p_clench = exp(beta*q_clench + beta4)/(exp(beta*q_pinch + beta3)+exp(beta*q_clench + beta4)+exp(beta*q_poke + beta5));
    p_poke = 1 - p_pinch - p_clench;
    
    a    = actions(t); % action on this trial
    o    = outcome(t); % outcome on this trial


    r = rand(1);

    if r < p_pinch
        p(t) = p_pinch;
        actions(t) = 1;
        o = out(t, 1);
        chosen_q(t) = q_pinch;
        chosen_aff(t) = aff_pinch;

        delta = o - q_pinch_;
        q(state, 1) = q(state, 1) + (alpha*delta);
        q(state, 2) = q(state, 2) - (alpha*delta/2);
        q(state, 3) = q(state, 3) - (alpha*delta/2);

        rpe_q = delta;
    elseif r < p_pinch + p_clench
        p(t) = p_clench;
        actions(t) = 2;
        o = out(t, 2);
        chosen_q(t) = q_clench;
        chosen_aff(t) = aff_clench;

        delta = o - q_clench_;
        q(state, 1) = q(state, 1) - (alpha*delta/2);
        q(state, 2) = q(state, 2) + (alpha*delta);
        q(state, 3) = q(state, 3) - (alpha*delta/2);

        rpe_q = delta;
    elseif r >= p_pinch + p_clench
        p(t) = p_poke;
        actions(t) = 3;
        o = out(t, 3);
        chosen_q(t) = q_poke;
        chosen_aff(t) = aff_poke;

        delta = o - q_poke_;
        q(state, 1) = q(state, 1) - (alpha*delta/2);
        q(state, 2) = q(state, 2) - (alpha*delta/2);
        q(state, 3) = q(state, 3) + (alpha*delta);
        
        rpe_q = delta;
    else
        p(t) = 1/3;
        o = 0;
    end
    rpe_q_list(t) = rpe_q;
    outcome(t) = o;
end

%===============================================================================
%===============================================================================

subj.reward_list = outcome;
subj.response_list = actions;
subj.gen_params = parameters;
subj.chosen_prob = p;
subj.chosen_q = chosen_q;
subj.chosen_aff = chosen_aff;
subj.rpe_q_list = rpe_q_list;
end
