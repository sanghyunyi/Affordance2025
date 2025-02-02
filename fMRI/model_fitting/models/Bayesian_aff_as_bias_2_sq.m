function [loglik] = model_RL(parameters,subj)
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

nd_r  = parameters(8);
r     = 1/(1+exp(-nd_r));

% unpack data
actions = subj.response_list; % 1 for action=f and 2 for action=j
outcome = subj.reward_list; % 1 for outcome=1 and 0 for outcome=0
states = subj.state_list; % stim idxs 
aff = subj.affordance_list; % pinch, clench, poke, palm, familiarity scores

% number of trials
T       = size(outcome,1);

% Bayesian update for each stim
b       = zeros(16*3, 6); % state, (n, r)

% to save probability of choice. Currently NaNs, will be filled below
p       = nan(T,1);


for t=1:T
    state = states(t);
    n_pinch = b(state, 1);
    r_pinch = b(state, 2);
    aff_pinch = (aff(t, 1)/(100))^2;
    beta_pinch = alpha1*(1-r)/r;

    n_clench = b(state, 3);
    r_clench = b(state, 4);
    aff_clench = (aff(t, 2)/(100))^2;
    beta_clench = alpha2*(1-r)/r;
    
    n_poke = b(state, 5);
    r_poke = b(state, 6);
    aff_poke = (aff(t, 3)/(100))^2;
    beta_poke = alpha3*(1-r)/r;

    q_pinch = (alpha1 + r_pinch)/(alpha1 + beta_pinch + n_pinch);
    q_clench = (alpha2 + r_clench)/(alpha2 + beta_clench + n_clench);
    q_poke = (alpha3 + r_poke)/(alpha3 + beta_poke + n_poke);


    p_pinch = exp(beta*q_pinch + beta2*aff_pinch)/(exp(beta*q_pinch + beta2*aff_pinch)+exp(beta*q_clench + beta2*aff_clench + beta3)+exp(beta*q_poke + beta2*aff_poke + beta4));
    p_clench = exp(beta*q_clench + beta2*aff_clench)/(exp(beta*q_pinch + beta2*aff_pinch - beta3)+exp(beta*q_clench + beta2*aff_clench)+exp(beta*q_poke + beta2*aff_poke + beta4 - beta3));
    p_poke = 1 - p_pinch - p_clench;
    
    % read info for the current trial
    a    = actions(t); % action on this trial
    o    = outcome(t); % outcome on this trial
    
    % store probability of the chosen action
    if a==1
        p(t) = p_pinch;
        b(state, 1) = n_pinch + 1;
        b(state, 2) = r_pinch + o;
    elseif a==2
        p(t) = p_clench;
        b(state, 3) = n_clench + 1;
        b(state, 4) = r_clench + o;
    elseif a==3
        p(t) = p_poke;
        b(state, 5) = n_poke + 1;
        b(state, 6) = r_poke + o;
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
