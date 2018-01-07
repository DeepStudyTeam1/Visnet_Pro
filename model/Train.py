import dataloader
from model import visnet_pro
import numpy as np
import torch
from torch.autograd import Variable

def train():
    model = visnet_pro()
    for param in model.parameters():
        if param.requires_grad == True:
            print(param.size())

train()