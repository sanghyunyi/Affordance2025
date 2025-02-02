function [loglik] = model_RL(parameters,subj)
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

nd_w_c  = parameters(6); % normally-distributed alpha
w_c     = 1/(1+exp(-nd_w_c)); % alpha (transformed to be between zero and one)

nd_cost = parameters(7);
cost = exp(nd_cost);

% unpack data
actions = subj.response_list; % 1 for action=f and 2 for action=j
outcome = subj.reward_list; % 1 for outcome=1 and 0 for outcome=0
states = subj.state_list; % stim idxs 
aff = subj.affordance_list; % pinch, clench, poke, palm, familiarity scores
is_new_block = subj.is_new_block;
shownCond = subj.shownCond; % incongruent/congruent high/low


% number of trials
T       = size(outcome,1);

% Q-value for each stim
q       = zeros(16*3, 3); % Q-value initialized at 0

% to save probability of choice. Currently NaNs, will be filled below
p       = nan(T,1);


for t=1:T
    state = states(t);
    q_pinch = q(state, 1);
    aff_pinch = aff(t, 1)/(100*beta2);
    q_clench = q(state, 2);
    aff_clench = aff(t, 2)/(100*beta2);
    q_poke = q(state, 3);
    aff_poke = aff(t, 3)/(100*beta2);

    p_pinch = exp(beta*q_pinch)/(exp(beta*q_pinch)+exp(beta*q_clench + beta3)+exp(beta*q_poke + beta4));
    p_clench = exp(beta*q_clench)/(exp(beta*q_pinch - beta3)+exp(beta*q_clench)+exp(beta*q_poke + beta4 - beta3));
    p_poke = 1 - p_pinch - p_clench;


    p_pinch_aff = exp(beta*aff_pinch)/(exp(beta*aff_pinch)+exp(beta*aff_clench+beta3)+exp(beta*aff_poke+beta4));
    p_clench_aff = exp(beta*aff_clench)/(exp(beta*aff_pinch-beta3)+exp(beta*aff_clench)+exp(beta*aff_poke+beta4-beta3));
    p_poke_aff = 1 - p_pinch_aff - p_clench_aff;


    %xi_d = abs(xi_q) - abs(xi_aff); % reliability difference, initialized at 0. xi_aff - xi_q    
    p_pinch = p_pinch*w_c + p_pinch_aff*(1-w_c);
    p_clench = p_clench*w_c + p_clench_aff*(1-w_c);
    p_poke = p_poke*w_c + p_poke_aff*(1-w_c);

    % read info for the current trial
    a    = actions(t); % action on this trial
    o    = outcome(t); % outcome on this trial

    % store probability of the chosen action
    if a==1
        p(t) = p_pinch;
        delta = o + cost*aff_pinch - q_pinch;
        q(state, 1) = q(state, 1) + (alpha*delta);

    elseif a==2
        p(t) = p_clench;
        delta = o + cost*aff_clench - q_clench;
        q(state, 2) = q(state, 2) + (alpha*delta);

    elseif a==3
        p(t) = p_poke;
        delta = o + cost*aff_poke - q_poke;
        q(state, 3) = q(state, 3) + (alpha*delta);

    else
        p(t) = 1/3;
    end
end

% log-likelihood is defined as the sum of log-probability of choice data 
% (given the parameters).
loglik = sum(log(p+eps));
% Note that eps is a very small number in matlab (type eps in the command 
% window to see how small it is), which does not have any effect in practice, 
% but it overcomes the problem of underflow when p is very very small 
% (effectively 0).
end
