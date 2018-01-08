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


# parameter
base_dir = os.path.split(os.getcwd())[0] + "/data/street2shop"

batch_size = 10
epochs = 10
learning_rate = 0.01

# create network
m1 = Visnet_Pro(light=True)
m2 = Tripletnet(m1)
if torch.cuda.is_available():
    m1 = m1.cuda()
    m2 = m2.cuda()

# create dataloader
transform = transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor()])
dataset = TripletImageLoader(base_path=base_dir + "/images",
                             triplets_file_path=base_dir + "/triplet/dresses/triplets.csv",
                             transform=transform)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

grad_param = []
for param in m2.parameters():
    if param.requires_grad == True:
        grad_param.append(param)
        print(param.size())

optimizer = torch.optim.Adam(grad_param, lr=learning_rate)

cudnn.benchmark = True

# train start
for i in range(1, epochs + 1):
    for batch_idx, (data1, data2, data3) in enumerate(train_loader):
        data1 = to_var(data1)
        data2 = to_var(data2)
        data3 = to_var(data3)

        loss = m2(data1, data2, data3)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 1 == 0:
            print('Epoch [%d/%d], Iter [%d/%d],  Loss: %.4f'
                  % (i , epochs, batch_idx, len(dataset) // batch_size, loss.data))

print("train success!")

torch.save(m2.state_dict(), 'm2.pkl')
