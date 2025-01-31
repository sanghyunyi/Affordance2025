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
beta3    = exp(nd_beta3);

nd_beta4 = parameters(5);
beta4    = nd_beta4;

nd_beta5 = parameters(6);
beta5    = nd_beta5;

nd_beta6 = parameters(7);
beta6    = nd_beta6;

nd_beta7 = parameters(8);
beta7    = nd_beta7;


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

c_list = nan(T,1);
wc_list = nan(T,1);
JSD_list = nan(T,1);
CPE_list = nan(T,1);
chosen_q_prob_list = nan(T, 1);
chosen_aff_prob_list = nan(T, 1);
rpe_q_list = nan(T,1);

c       = zeros(16*3, 1); % conflict, initialized at 0

for t=1:T
    state = states(t);
    q_pinch = q(state, 1);
    aff_pinch = aff(t, 1)/(100);
    q_clench = q(state, 2);
    aff_clench = aff(t, 2)/(100);
    q_poke = q(state, 3);
    aff_poke = aff(t, 3)/(100);

    p_pinch = exp(beta*q_pinch+beta5)/(exp(beta*q_pinch+beta5)+exp(beta*q_clench+beta6)+exp(beta*q_poke+beta7)) + 1e-64;
    p_clench = exp(beta*q_clench+beta6)/(exp(beta*q_pinch+beta5)+exp(beta*q_clench+beta6)+exp(beta*q_poke+beta7)) + 1e-64;
    p_poke = exp(beta*q_poke+beta7)/(exp(beta*q_pinch+beta5)+exp(beta*q_clench+beta6)+exp(beta*q_poke+beta7)) + 1e-64;
    
    p_pinch_aff = exp(beta2*aff_pinch+beta5)/(exp(beta2*aff_pinch+beta5)+exp(beta2*aff_clench+beta6)+exp(beta2*aff_poke+beta7)) + 1e-64;
    p_clench_aff = exp(beta2*aff_clench+beta6)/(exp(beta2*aff_pinch+beta5)+exp(beta2*aff_clench+beta6)+exp(beta2*aff_poke+beta7)) + 1e-64;
    p_poke_aff = exp(beta2*aff_poke+beta7)/(exp(beta2*aff_pinch+beta5)+exp(beta2*aff_clench+beta6)+exp(beta2*aff_poke+beta7)) + 1e-64;

    
    JSD = (0.5*(p_pinch*log(p_pinch/p_pinch_aff) + p_clench*log(p_clench/p_clench_aff) + p_poke*log(p_poke/p_poke_aff) + p_pinch_aff*log(p_pinch_aff/p_pinch) + p_clench_aff*log(p_clench_aff/p_clench) + p_poke_aff*log(p_poke_aff/p_poke)))^0.5;

    c(state) = JSD;

    w_c = 1/(1+exp(-beta3*c(state)+beta4));
    
    p_pinch_ = p_pinch*w_c + p_pinch_aff*(1-w_c);
    p_clench_ = p_clench*w_c + p_clench_aff*(1-w_c);
    p_poke_ = p_poke*w_c + p_poke_aff*(1-w_c);
    
    
    a    = actions(t); % action on this trial
    o    = outcome(t); % outcome on this trial

    r = rand(1);
    if r < p_pinch
        p(t) = p_pinch_;
        actions(t) = 1;
        o = out(t, 1);
        chosen_q_prob_list(t) = p_pinch;
        chosen_aff_prob_list(t) = p_pinch_aff;
        delta = o - q_pinch;
        q(state, 1) = q(state, 1) + (alpha*delta);
        q(state, 2) = q(state, 2) - (alpha*delta/2);
        q(state, 3) = q(state, 3) - (alpha*delta/2);
        rpe_q = delta;
    elseif r < p_pinch + p_clench
        p(t) = p_clench_;
        actions(t) = 2;
        o = out(t, 2);
        chosen_q_prob_list(t) = p_clench;
        chosen_aff_prob_list(t) = p_clench_aff;
        delta = o - q_clench;
        q(state, 1) = q(state, 1) - (alpha*delta/2);
        q(state, 2) = q(state, 2) + (alpha*delta);
        q(state, 3) = q(state, 3) - (alpha*delta/2);
        rpe_q = delta;
    elseif r >= p_pinch + p_clench
        p(t) = p_poke_;
        actions(t) = 3;
        o = out(t, 3);
        chosen_q_prob_list(t) = p_poke;
        chosen_aff_prob_list(t) = p_poke_aff;
        delta = o - q_poke;
        q(state, 1) = q(state, 1) - (alpha*delta/2);
        q(state, 2) = q(state, 2) - (alpha*delta/2);
        q(state, 3) = q(state, 3) + (alpha*delta);
        rpe_q = delta;
    else
        p(t) = 1/3;
    end
    outcome(t) = o;
    wc_list(t) = w_c;
    c_list(t) = c(state);
    JSD_list(t) = JSD;
    %CPE_list(t) = cpe;
    rpe_q_list(t) = rpe_q;
end

%===============================================================================
%===============================================================================

subj.reward_list = outcome;
subj.response_list = actions;
subj.gen_params = parameters;
subj.wc_list = wc_list;
subj.c_list = c_list;
subj.JSD_list = JSD_list;
subj.CPE_list = CPE_list;
subj.chosen_q_prob = chosen_q_prob_list;
subj.chosen_aff_prob = chosen_aff_prob_list;
subj.chosen_prob = p;
subj.rpe_q_list = rpe_q_list;

end
