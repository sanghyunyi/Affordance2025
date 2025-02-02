function [subj] = model_RL(parameters,subj)
% lr for slot and cue
nd_alpha  = parameters(1); % normally-distributed alpha
alpha     = 1/(1+exp(-nd_alpha)); % alpha (transformed to be between zero and one)

nd_alpha_xi  = parameters(2); % normally-distributed alpha
alpha_xi     = 1/(1+exp(-nd_alpha_xi)); % alpha (transformed to be between zero and one)

% inverse temp for cue
nd_beta = parameters(3);
beta    = exp(nd_beta);

nd_beta2 = parameters(4);
beta2    = exp(nd_beta2)+1;

nd_beta3 = parameters(5);
beta3    = exp(nd_beta3);

nd_beta4 = parameters(6);
beta4    = nd_beta4;

nd_beta5 = parameters(7);
beta5    = exp(nd_beta5);

nd_beta6 = parameters(8);
beta6    = nd_beta6;

nd_beta7 = parameters(9);
beta7    = nd_beta7;

nd_beta8 = parameters(10);
beta8    = nd_beta8;

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

wc_list = nan(T,1);
xi_aff_list = nan(T,1);
xi_q_list = nan(T,1);
rpe_aff_list = nan(T,1);
rpe_q_list = nan(T,1);

xi_aff = zeros(16*3, 1);
xi_q = zeros(16*3, 1);

for t=1:T
    state = states(t);
    q_pinch = q(state, 1);
    aff_pinch = aff(t, 1)/(100*beta2);
    q_clench = q(state, 2);
    aff_clench = aff(t, 2)/(100*beta2);
    q_poke = q(state, 3);
    aff_poke = aff(t, 3)/(100*beta2);

    p_pinch = exp(beta*q_pinch+beta6)/(exp(beta*q_pinch+beta6)+exp(beta*q_clench+beta7)+exp(beta*q_poke+beta8));
    p_clench = exp(beta*q_clench+beta7)/(exp(beta*q_pinch+beta6)+exp(beta*q_clench+beta7)+exp(beta*q_poke+beta8));
    p_poke = 1 - p_pinch - p_clench;

    p_pinch_aff = exp(beta*aff_pinch+beta6)/(exp(beta*aff_pinch+beta6)+exp(beta*aff_clench+beta7)+exp(beta*aff_poke+beta8));
    p_clench_aff = exp(beta*aff_clench+beta7)/(exp(beta*aff_pinch+beta6)+exp(beta*aff_clench+beta7)+exp(beta*aff_poke+beta8));
    p_poke_aff = 1 - p_pinch_aff - p_clench_aff;
    
    %xi_d = abs(xi_q) - abs(xi_aff); % reliability difference, initialized at 0. xi_aff - xi_q
    xi_aff_list(t) = xi_aff(state);
    xi_q_list(t) = xi_q(state);

    w_c = 1/(1+exp(-beta3*abs(xi_q(state))+beta5*abs(xi_aff(state)) + beta4));
    
    wc_list(t) = w_c;
    
    p_pinch = p_pinch*w_c + p_pinch_aff*(1-w_c);
    p_clench = p_clench*w_c + p_clench_aff*(1-w_c);
    p_poke = p_poke*w_c + p_poke_aff*(1-w_c);
    
    r = rand(1);


    if r < p_pinch
        p(t) = p_pinch;
        actions(t) = 1;
        o = out(t, 1);
        chosen_q(t) = q_pinch;
        chosen_aff(t) = aff_pinch;

        delta = o - q_pinch;
        q(state, 1) = q(state, 1) + (alpha*delta);
        q(state, 2) = q(state, 2) - (alpha*delta/2);
        q(state, 3) = q(state, 3) - (alpha*delta/2);

        rpe_q = delta;
        xi_q(state) = xi_q(state) + alpha_xi*((1-abs(rpe_q))-xi_q(state));
        rpe_aff = o - aff_pinch;
        xi_aff(state) = xi_aff(state) + alpha_xi*((1-abs(rpe_aff))-xi_aff(state));
    elseif r < p_pinch + p_clench 
        p(t) = p_clench;
        actions(t) = 2;
        o = out(t, 2);
        chosen_q(t) = q_clench;
        chosen_aff(t) = aff_clench;

        delta = o - q_clench;
        q(state, 1) = q(state, 1) - (alpha*delta/2);
        q(state, 2) = q(state, 2) + (alpha*delta);
        q(state, 3) = q(state, 3) - (alpha*delta/2);

        rpe_q = delta;
        xi_q(state) = xi_q(state) + alpha_xi*((1-abs(rpe_q))-xi_q(state));
        rpe_aff = o - aff_clench;
        xi_aff(state) = xi_aff(state) + alpha_xi*((1-abs(rpe_aff))-xi_aff(state));
    elseif r >= p_pinch + p_clench
        p(t) = p_poke;
        actions(t) = 3;
        o = out(t, 3);
        chosen_q(t) = q_poke;
        chosen_aff(t) = aff_poke;

        delta = o - q_poke;
        q(state, 1) = q(state, 1) - (alpha*delta/2);
        q(state, 2) = q(state, 2) - (alpha*delta/2);
        q(state, 3) = q(state, 3) + (alpha*delta);

        rpe_q = delta;
        xi_q(state) = xi_q(state) + alpha_xi*((1-abs(rpe_q))-xi_q(state));
        rpe_aff = o - aff_poke;
        xi_aff(state) = xi_aff(state) + alpha_xi*((1-abs(rpe_aff))-xi_aff(state));
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
subj.chosen_q = chosen_q;
subj.chosen_aff = chosen_aff;
subj.wc_list = wc_list;
subj.xi_aff_list = xi_aff_list;
subj.xi_q_list = xi_q_list;
subj.rpe_aff_list = rpe_aff_list;
subj.rpe_q_list = rpe_q_list;
end
