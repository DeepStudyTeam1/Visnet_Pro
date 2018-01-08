import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable


class Visnet_Pro(nn.Module):
    def __init__(self, light=False):
        print("Create Visnet_Pro")
        super(Visnet_Pro, self).__init__()
        self.layer1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=5, stride=5, padding=1),  # (?,3,60,60)
            nn.Conv2d(3, 128, kernel_size=6, stride=4, padding=1),  # (?,128,15,15)
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=7, stride=4, padding=0),  # (?,128,3,3)
        )
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=10, stride=10, padding=1),  # (?,3,30,30)
            nn.Conv2d(3, 128, kernel_size=6, stride=4, padding=0),  # (?,128,7,7)
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),  # (?,128,3,3)
        )

        self.bn1 = nn.BatchNorm1d(2304)

        self.inception = torchvision.models.inception_v3(pretrained=True)
        for param in self.inception.parameters():
            param.requires_grad = False
        if light == False:
            self.inception.fc = nn.Linear(2048, 4096)
            self.bn2 = nn.BatchNorm1d(4096)
            self.fc1 = nn.Linear(6400, 4096)
            self.bn3 = nn.BatchNorm1d(4096)
        else :
            self.inception.fc = nn.Linear(2048, 2048)
            self.bn2 = nn.BatchNorm1d(2048)
            self.fc1 = nn.Linear(4352, 2048)
            self.bn3 = nn.BatchNorm1d(2048)

    def forward(self, x):
        print(x.size())
        out1 = self.layer1(x)  # (?, 128,3,3)
        print(out1.size())
        out1 = out1.view(out1.size(0), -1)  # (?, 1152)
        print(out1.size())
        out2 = self.layer2(x)  # (?,128,3,3)
        print(out2.size())
        out2 = out2.view(out2.size(0), -1)  # (?, 1152)
        print(out2.size())

        cat1 = torch.cat((out1, out2), dim=1)  # (?,1152)
        print(cat1.size())
        norm1 = self.bn1(cat1)
        print(norm1.size())

        out3, _ = self.inception(x)
        print(out3.size())
        norm2 = self.bn2(out3)
        print(norm2.size())

        cat2 = torch.cat((norm1, norm2), dim=1)
        print(cat2.size())

        fc1 = self.fc1(cat2)
        print(fc1.size())
        out = self.bn3(fc1)
        print(out.size())
        return out


class Tripletnet(nn.Module):
    def __init__(self, embeddingnet):
        super(Tripletnet, self).__init__()
        self.embeddingnet = embeddingnet

    def forward(self, x, y, z):
        embedded_x = self.embeddingnet(x)
        embedded_y = self.embeddingnet(y)
        embedded_z = self.embeddingnet(z)
        dist_a = F.pairwise_distance(embedded_x, embedded_y, 2)
        dist_b = F.pairwise_distance(embedded_x, embedded_z, 2)
        return dist_a, dist_b, embedded_x, embedded_y, embedded_z
