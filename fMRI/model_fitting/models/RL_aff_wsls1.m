function [loglik] = model_RL(parameters,subj)
% lr for slot and cue
%nd_alpha  = parameters(1); % normally-distributed alpha
%alpha     = 1/(1+exp(-nd_alpha)); % alpha (transformed to be between zero and one)

% inverse temp for cue
nd_beta = parameters(1);
beta    = exp(nd_beta);

%nd_beta2 = parameters(3);
%beta2    = exp(nd_beta2);

% unpack data
actions = subj.response_list; % 1 for action=f and 2 for action=j
outcome = subj.reward_list; % 1 for outcome=1 and 0 for outcome=0
states = subj.state_list; % stim idxs 
aff = subj.affordance_list; % pinch, clench, poke, palm, familiarity scores

% number of trials
T       = size(outcome,1);

% Q-value for each stim
%q       = zeros(16*3, 3); % Q-value initialized at 0

% to save probability of choice. Currently NaNs, will be filled below
p       = nan(T,1);

last_actions = zeros(16*3, 1);
last_outcomes = zeros(16*3, 1);

for t=1:T
    state = states(t);
    
    %q_pinch = q(state, 1);
    aff_pinch = aff(t, 1)/(100);
    %q_clench = q(state, 2);
    aff_clench = aff(t, 2)/(100);
    %q_poke = q(state, 3);
    aff_poke = aff(t, 3)/(100);

    if last_actions(state) == 0
        [~, action] = max([aff_pinch, aff_clench, aff_poke]);
        if action == 1
            q_pinch = 1;
            q_clench = 0;
            q_poke = 0;
        elseif action == 2
            q_pinch = 0;
            q_clench = 1;
            q_poke = 0;
        else
            q_pinch = 0;
            q_clench = 0;
            q_poke = 1;
        end

    else
        % Retrieve last action and outcome for this state
        last_action = last_actions(state);
        last_outcome = last_outcomes(state);
        
        % Win-Stay Lose-Shift Logic
        if last_outcome > 0  % Win condition
            action = last_action;  % Stay with the last action
            
            if action == 1
                q_pinch = 1;
                q_clench = 0;
                q_poke = 0;
            elseif action == 2
                q_pinch = 0;
                q_clench = 1;
                q_poke = 0;
            else
                q_pinch = 0;
                q_clench = 0;
                q_poke = 1;
            end

        else  % Lose condition
            % Shift to a different action
            %possible_actions = setdiff([1, 2, 3], last_action);
            %action = possible_actions(randi(length(possible_actions)));
            if last_action == 1
                q_pinch = 0;
                q_clench = 0.5;
                q_poke = 0.5;
            elseif last_action == 2
                q_pinch = 0.5;
                q_clench = 0;
                q_poke = 0.5;
            else
                q_pinch = 0.5;
                q_clench = 0.5;
                q_poke = 0;
            end
        end
    end
    
    p_pinch = exp(beta*q_pinch)/(exp(beta*q_pinch)+exp(beta*q_clench)+exp(beta*q_poke));
    p_clench = exp(beta*q_clench)/(exp(beta*q_pinch)+exp(beta*q_clench)+exp(beta*q_poke));
    p_poke = 1 - p_pinch - p_clench;

    % read info for the current trial
    a    = actions(t); % action on this trial
    o    = outcome(t); % outcome on this trial
    
    % store probability of the chosen action
    if a==1
        p(t) = p_pinch;
        %delta = o - q(state, 1);
        %q(state, 1) = q(state, 1) + (alpha*delta);
    elseif a==2
        p(t) = p_clench;
        %delta = o - q(state, 2);
        %q(state, 2) = q(state, 2) + (alpha*delta);
    elseif a==3
        p(t) = p_poke;
        %delta = o - q(state, 3);
        %q(state, 3) = q(state, 3) + (alpha*delta);
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
