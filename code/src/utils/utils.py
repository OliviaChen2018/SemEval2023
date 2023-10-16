import os
import random
import torch
import numpy as np
import argparse
import torch.nn.functional as F
import torch
from torch import nn

def set_random_seed(random_seed):
    """Set random seed"""
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    torch.backends.cudnn.deterministic = True
        
def freeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = False
            
def unfreeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = True   
            
def compute_kl_loss(p, q, pad_mask=None):
    
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    
    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss

class NoisyTune:

    def __init__(self, model: nn.Module):

        self.model = model
    
    def add_noisy(self, noise_lambda=0.2):
        for name ,para in self.model.named_parameters():
            self.model.state_dict()[name][:] += (torch.rand(para.size()) - 0.5) * noise_lambda * torch.std(para)