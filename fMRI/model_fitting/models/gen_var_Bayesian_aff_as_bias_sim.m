function [subj] = model_RL(parameters,subj)
% lr for slot and cue
nd_alpha1  = parameters(1);
alpha1     = exp(nd_alpha1);

nd_alpha2  = parameters(2);
alpha2     = exp(nd_alpha2);

nd_alpha3  = parameters(3);
alpha3     = exp(nd_alpha3);

% inverse temp for cue
nd_beta = parameters(4);
beta    = exp(nd_beta);

nd_beta2 = parameters(5);
beta2    = exp(nd_beta2);

nd_beta3 = parameters(6);
beta3    = nd_beta3;

nd_beta4 = parameters(7);
beta4    = nd_beta4;

nd_beta5 = parameters(8);
beta5    = nd_beta5;

nd_r  = parameters(9);
r     = 1/(1+exp(-nd_r));

% unpack data
out = subj.out_list; % 1 for outcome=1 and 0 for outcome=0
aff = subj.affordance_list; % pinch, clench, poke, palm, familiarity scores
states = subj.state_list; % stim idxs 
is_new_block = subj.is_new_block;


% number of trials
T       = size(out,1);

% Bayesian update for each stim
b       = zeros(16*3, 6); % state, (n, r)

% to save probability of choice. Currently NaNs, will be filled below
p       = nan(T,1);
outcome = nan(T,1);
actions = nan(T,1);

chosen_aff = nan(T, 1);
chosen_q = nan(T, 1);

rpe_q_list = nan(T,1);

for t=1:T
    state = states(t);
    n_pinch = b(state, 1);
    r_pinch = b(state, 2);
    aff_pinch = aff(t, 1)/(100);
    beta_pinch = alpha1*(1-r)/r;

    n_clench = b(state, 3);
    r_clench = b(state, 4);
    aff_clench = aff(t, 2)/(100);
    beta_clench = alpha2*(1-r)/r;
    
    n_poke = b(state, 5);
    r_poke = b(state, 6);
    aff_poke = aff(t, 3)/(100);
    beta_poke = alpha3*(1-r)/r;

    q_pinch = (alpha1 + r_pinch)/(alpha1 + beta_pinch + n_pinch);
    q_clench = (alpha2 + r_clench)/(alpha2 + beta_clench + n_clench);
    q_poke = (alpha3 + r_poke)/(alpha3 + beta_poke + n_poke);


    p_pinch = exp(beta*q_pinch + beta2*aff_pinch + beta3)/(exp(beta*q_pinch + beta2*aff_pinch + beta3)+exp(beta*q_clench + beta2*aff_clench + beta4)+exp(beta*q_poke + beta2*aff_poke + beta5));
    p_clench = exp(beta*q_clench + beta2*aff_clench + beta4)/(exp(beta*q_pinch + beta2*aff_pinch + beta3)+exp(beta*q_clench + beta2*aff_clench + beta4)+exp(beta*q_poke + beta2*aff_poke + beta5));
    p_poke = 1 - p_pinch - p_clench;
    
    r = rand(1);

    if r < p_pinch
        p(t) = p_pinch;
        actions(t) = 1;
        o = out(t, 1);
        chosen_q(t) = q_pinch;
        chosen_aff(t) = aff_pinch;

        b(state, 1) = n_pinch + 1;
        b(state, 2) = r_pinch + o;
    elseif r < p_pinch + p_clench
        p(t) = p_clench;
        actions(t) = 2;
        o = out(t, 2);
        chosen_q(t) = q_clench;
        chosen_aff(t) = aff_clench;

        b(state, 3) = n_clench + 1;
        b(state, 4) = r_clench + o;
    elseif r >= p_pinch + p_clench
        p(t) = p_poke;
        actions(t) = 3;
        o = out(t, 3);
        chosen_q(t) = q_poke;
        chosen_aff(t) = aff_poke;
        
        b(state, 5) = n_poke + 1;
        b(state, 6) = r_poke + o;
    else
        p(t) = 1/3;
        o = 0;
    end
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
