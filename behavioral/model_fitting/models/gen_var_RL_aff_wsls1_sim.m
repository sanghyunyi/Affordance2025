function [subj] = model_RL(parameters,subj)
% lr for slot and cue
%nd_alpha  = parameters(1); % normally-distributed alpha
%alpha     = 1/(1+exp(-nd_alpha)); % alpha (transformed to be between zero and one)

% inverse temp for cue
nd_beta = parameters(1);
beta    = exp(nd_beta);

%nd_beta2 = parameters(3);
%beta2    = exp(nd_beta2);

% unpack data
%actions = subj.response_list; % 1 for action=f and 2 for action=j # from subj
%outcome = subj.reward_list; % 1 for outcome=1 and 0 for outcome=0 # from subj
out = subj.out_list; % 1 for outcome=1 and 0 for outcome=0
aff = subj.affordance_list; % pinch, clench, poke, palm, familiarity scores
states = subj.state_list; % stim idxs 
is_new_block = subj.is_new_block;

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

    r = rand(1);

    if r < p_pinch
        p(t) = p_pinch;
        actions(t) = 1;
        o = out(t, 1);
    elseif r < p_pinch + p_clench
        p(t) = p_clench;
        actions(t) = 2;
        o = out(t, 2);
    elseif r >= p_pinch + p_clench
        p(t) = p_poke;
        actions(t) = 3;
        o = out(t, 3);
    else
        p(t) = 1/3;
        o = 0;
    end
    outcome(t) = o;
end

subj.reward_list = outcome;
subj.response_list = actions;
subj.gen_params = parameters;
subj.chosen_prob = p;
subj.chosen_q = chosen_q;
subj.chosen_aff = chosen_aff;
subj.rpe_q_list = rpe_q_list;
end
