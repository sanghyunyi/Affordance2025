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
beta3    = exp(nd_beta3);

nd_beta4 = parameters(5);
beta4    = nd_beta4;

nd_beta6 = parameters(6);
beta6    = nd_beta6;

nd_beta7 = parameters(7);
beta7    = nd_beta7;

nd_beta8 = parameters(8);
beta8    = nd_beta8;

% unpack data
actions = subj.response_list; % 1 for action=f and 2 for action=j
outcome = subj.reward_list; % 1 for outcome=1 and 0 for outcome=0
states = subj.state_list; % stim idxs 
aff = subj.affordance_list; % pinch, clench, poke, palm, familiarity scores
is_new_block = subj.is_new_block;

% number of trials
T       = size(outcome,1);

% Q-value for each stim
q       = zeros(16*3, 3); % Q-value initialized at 0

% to save probability of choice. Currently NaNs, will be filled below
p       = nan(T,1);

b = zeros(16*3, 2);

for t=1:T
    state = states(t);
    q_pinch = q(state, 1);
    aff_pinch = aff(t, 1)/(100*beta2);
    q_clench = q(state, 2);
    aff_clench = aff(t, 2)/(100*beta2);
    q_poke = q(state, 3);
    aff_poke = aff(t, 3)/(100*beta2);

    I_aff = b(state, 1);
    I_nonaff = b(state, 2);

    p_pinch = exp(beta*q_pinch+beta6)/(exp(beta*q_pinch+beta6)+exp(beta*q_clench+beta7)+exp(beta*q_poke+beta8));
    p_clench = exp(beta*q_clench+beta7)/(exp(beta*q_pinch+beta6)+exp(beta*q_clench+beta7)+exp(beta*q_poke+beta8));
    p_poke = exp(beta*q_poke+beta8)/(exp(beta*q_pinch+beta6)+exp(beta*q_clench+beta7)+exp(beta*q_poke+beta8));

    p_q = [p_pinch p_clench p_poke];
    p_q(isnan(p_q)) = 1;


    p_pinch_aff = exp(beta*aff_pinch+beta6)/(exp(beta*aff_pinch+beta6)+exp(beta*aff_clench+beta7)+exp(beta*aff_poke+beta8));
    p_clench_aff = exp(beta*aff_clench+beta7)/(exp(beta*aff_pinch+beta6)+exp(beta*aff_clench+beta7)+exp(beta*aff_poke+beta8));
    p_poke_aff = exp(beta*aff_poke+beta8)/(exp(beta*aff_pinch+beta6)+exp(beta*aff_clench+beta7)+exp(beta*aff_poke+beta8));
    
    p_aff = [p_pinch_aff p_clench_aff p_poke_aff];
    p_aff(isnan(p_aff)) = 1;

    aff_xi = I_aff;
    nonaff_xi = I_nonaff;

    w_c = 1/(1+exp(-beta3*(nonaff_xi - aff_xi) + beta4));
    
    p_all = p_q*w_c + p_aff*(1-w_c);
    
    p_pinch = p_all(1);
    p_clench = p_all(2);
    p_poke = p_all(3);
    
    % read info for the current trial
    a    = actions(t); % action on this trial
    o    = outcome(t); % outcome on this trial

    b(state, 1) = (1-alpha)*I_aff + alpha*o*p_aff(a)/(p_all(a)+1e-64);
    b(state, 2) = (1-alpha)*I_nonaff + alpha*o*p_q(a)/(p_all(a)+1e-64);

    % store probability of the chosen action
    if a==1
        p(t) = p_pinch;
        delta = o - q_pinch;
        q(state, 1) = q(state, 1) + (alpha*delta);
        q(state, 2) = q(state, 2) - (alpha*delta/2);
        q(state, 3) = q(state, 3) - (alpha*delta/2);
    elseif a==2
        p(t) = p_clench;
        delta = o - q_clench;
        q(state, 1) = q(state, 1) - (alpha*delta/2);
        q(state, 2) = q(state, 2) + (alpha*delta);
        q(state, 3) = q(state, 3) - (alpha*delta/2);
    elseif a==3
        p(t) = p_poke;
        delta = o - q_poke;
        q(state, 1) = q(state, 1) - (alpha*delta/2);
        q(state, 2) = q(state, 2) - (alpha*delta/2);
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
