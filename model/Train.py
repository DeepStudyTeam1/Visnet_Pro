from loader import TripletImageLoader
from model import Visnet_Pro, Tripletnet
import argparse
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np


def to_var(x):
    """Convert tensor to variable."""
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def train():
    base_dir = "C:/Users/cksdn/Documents/GitHub/Visnet_Pro/data/street2shop"
    batch_size = 5

    m1 = Visnet_Pro(light=True)
    m2 = Tripletnet(m1)
    if torch.cuda.is_available():
        m2 = m2.cuda()

    transform = transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor()])
    dataset = TripletImageLoader(base_path=base_dir + "/images",
                                 triplets_file_path=base_dir + "/triplet/dresses/triplets.csv",
                                 transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for batch_idx, (data1, data2, data3) in enumerate(train_loader):
        data1 = to_var(data1)
        data2 = to_var(data2)
        data3 = to_var(data3)




train()
