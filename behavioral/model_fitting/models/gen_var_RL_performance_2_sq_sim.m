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

nd_beta6 = parameters(6);
beta6    = nd_beta6;

nd_beta7 = parameters(7);
beta7    = nd_beta7;


% unpack data
%actions = subj.response_list; % 1 for action=f and 2 for action=j # from subj
%outcome = subj.reward_list; % 1 for outcome=1 and 0 for outcome=0 # from subj
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
chosen_p   = nan(T, 1);
chosen_aff = nan(T, 1);
chosen_q = nan(T, 1);

all_aff = nan(T, 6);
all_q = nan(T, 6);

wc_list = nan(T,1);
xi_aff_list = nan(T,1);
xi_q_list = nan(T,1);
n_aff_list = nan(T,1);
r_aff_list = nan(T,1);
n_q_list = nan(T,1);
r_q_list = nan(T,1);

b = zeros(16*3, 2);

outcome = nan(T,1);
actions = nan(T,1);

for t=1:T
    state = states(t);
    q_pinch = q(state, 1);
    aff_pinch = (aff(t, 1)/(100))^2/beta2;
    q_clench = q(state, 2);
    aff_clench = (aff(t, 2)/(100))^2/beta2;
    q_poke = q(state, 3);
    aff_poke = (aff(t, 3)/(100))^2/beta2;

    I_aff = b(state, 1);
    I_nonaff = b(state, 2);

    p_pinch = exp(beta*q_pinch)/(exp(beta*q_pinch)+exp(beta*q_clench + beta6)+exp(beta*q_poke + beta7));
    p_clench = exp(beta*q_clench)/(exp(beta*q_pinch - beta6)+exp(beta*q_clench)+exp(beta*q_poke + beta7 - beta6));
    p_poke = 1 - p_pinch - p_clench;

    p_q = [p_pinch p_clench p_poke];
    p_q(isnan(p_q)) = 1;


    p_pinch_aff = exp(beta*aff_pinch)/(exp(beta*aff_pinch)+exp(beta*aff_clench+beta6)+exp(beta*aff_poke+beta7));
    p_clench_aff = exp(beta*aff_clench)/(exp(beta*aff_pinch-beta6)+exp(beta*aff_clench)+exp(beta*aff_poke+beta7-beta6));
    p_poke_aff = 1 - p_pinch_aff - p_clench_aff;

    
    p_aff = [p_pinch_aff p_clench_aff p_poke_aff];
    p_aff(isnan(p_aff)) = 1;
    
    aff_xi = I_aff;
    nonaff_xi = I_nonaff;

    w_c = 1/(1+exp(-beta3*(nonaff_xi - aff_xi) + beta4));
    
    xi_aff_list(t) = aff_xi;
    xi_q_list(t) = nonaff_xi;
    
    wc_list(t) = w_c;

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

    p_all = [p_pinch p_clench p_poke];
    
    r = rand(1);
    
    if r < p_pinch
        chosen_p(t) = p_pinch;
        chosen_q(t) = q_pinch;
        chosen_aff(t) = aff_pinch;
        actions(t) = 1;
        o = out(t, 1);
        
        delta = o - q_pinch;
        q(state, 1) = q(state, 1) + (alpha*delta);

    elseif r < p_pinch + p_clench 
        chosen_p(t) = p_clench;
        chosen_q(t) = q_clench;
        chosen_aff(t) = aff_clench;
        actions(t) = 2;
        o = out(t, 2);
        
        delta = o - q_clench;
        q(state, 2) = q(state, 2) + (alpha*delta);

    elseif r >= p_pinch + p_clench
        chosen_p(t) = p_poke;
        chosen_q(t) = q_poke;
        chosen_aff(t) = aff_poke;
        actions(t) = 3;
        o = out(t, 3);
        
        delta = o - q_poke;
        q(state, 3) = q(state, 3) + (alpha*delta);

    else
        chosen_p(t) = 1/3;
        o = 0;
    end
    
    b(state, 1) = (1-alpha)*I_aff + alpha*o*p_aff(actions(t))/(p_all(actions(t))+1e-64);
    b(state, 2) = (1-alpha)*I_nonaff + alpha*o*p_q(actions(t))/(p_all(actions(t))+1e-64);

    outcome(t) = o;
end

%===============================================================================
%===============================================================================

subj.reward_list = outcome;
subj.response_list = actions;
subj.gen_params = parameters;
subj.chosen_prob = chosen_p;
subj.chosen_q = chosen_q;
subj.all_q = all_q;
subj.chosen_aff = chosen_aff;
subj.all_aff = all_aff;
subj.wc_list = wc_list;
subj.xi_aff_list = xi_aff_list;
subj.xi_q_list = xi_q_list;
subj.n_aff_list = n_aff_list;
subj.r_aff_list = r_aff_list;
subj.n_q_list = n_q_list;
subj.r_q_list = r_q_list;
end
