import numpy as np
from scipy.stats import norm, beta, uniform
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import autograd

# model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.weight = torch.tensor(np.zeros((24*2, 3)), requires_grad=True) #nn.Linear(72 + 12, 1, bias=False)
        self.prev_action = None
        self.prev_selected_stim = None
        self.prev_won = None # for win stay lose shift. let it stick to the prev stim. This can be modeled also as high learning rate.

    def forward(self, x):
        # x should be a one hot vector
        return self.weight[x]


def actor(model, state, aff, beta_list):
    """
    Args are the inputs to the model, besides the general model params:
    Args:
        Q: the expected action value, computed by learner (for all choices; vector)
        beta: softmax inverse temperature; vector
    """
    beta = beta_list[0]
    aff = torch.tensor(aff)
    # Action selection through logistic function
    pOptions = F.softmax(beta * (model(state)+aff))

    ## Note: if left response is assigned to index 0 and right to index 1 in [selectedIndex]
    ## (for taskStruct['Q']), then this returns p(right choice)
    ## Pick an option given the softmax probabilities
    #print(pOptions)
    cumsum = torch.cumsum(pOptions, dim=0)
    cumsum[-1] = 1.0
    respIdx = np.where(cumsum.detach().numpy() >= np.random.rand(1))[0][0]
    # output: 0 means left choice, 1 means right choice
    return(respIdx, pOptions)


class fitParamContain():
    def __init__(self, fitlikelihood):
        self.fitlikelihood = fitlikelihood
        return

    def instr_deltaLearner_fitParams(self, fitParams):
        self.alpha_i = fitParams[0]
        self.beta_i = fitParams[1]
        self.beta0_i = fitParams[2]
        self.beta1_i = fitParams[3]
        self.beta2_i = fitParams[4]
        return self

class genParamContain():
    def __init__(self):
        return

    def instr_deltaLearner_genParams(self, currParams):
        self.alpha_i = currParams[0]
        self.beta_i = currParams[1]
        self.beta0_i = currParams[2]
        self.beta1_i = currParams[3]
        self.beta2_i = currParams[4]
        return self

