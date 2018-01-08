from loader import TripletImageLoader
from model import Visnet_Pro, Tripletnet
import numpy as np
import torch
from torch.autograd import Variable


def to_var(x):
    """Convert tensor to variable."""
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def train():
    m1 = Visnet_Pro(light = True)
    m2 = Tripletnet(m1)
    for param in m1.parameters():
        if param.requires_grad == True:
            print(param.size())

    test = torch.randn(2,3,299,299)
    test = to_var(test)
    out = m1(test)
    print(out)





train()