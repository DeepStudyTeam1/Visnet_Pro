import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class LRN(nn.Module):
    def __init__ (self):
        super (LRN, self).__init__ ()
    def forward (self, x):
        div = x.pow (2)
        div = div.sum(1, keepdim = True)
        div = div.expand(x.size())
        div = div.add (1.0).pow (0.5)
        x = x.div(div)
        return x

class Visnet_Pro(nn.Module):
    def __init__(self, heavy=False):
        print("Create Visnet_Pro!")
        super(Visnet_Pro, self).__init__()
        self.layer1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=5, stride=5, padding=1),  # (?,3,60,60)
            nn.Conv2d(3, 64, kernel_size=6, stride=4, padding=1),  # (?,64,15,15)
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=7, stride=4, padding=0),  # (?,64,3,3)
        )
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=10, stride=10, padding=1),  # (?,3,30,30)
            nn.Conv2d(3, 64, kernel_size=6, stride=4, padding=0),  # (?,64,7,7)
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),  # (?,64,3,3)
        )

        self.lrn = LRN()

        self.inception = torchvision.models.inception_v3(pretrained=True)
        for param in self.inception.parameters():
            param.requires_grad = False
        if heavy == True:
            self.inception.fc = nn.Linear(2048, 4096)
            self.fc1 = nn.Linear(6400, 4096)
        else :
            self.inception.fc = nn.Linear(2048, 1024)
            self.fc1 = nn.Linear(2176, 1024)

    def forward(self, x):
        out1 = self.layer1(x)  # (?, 128,3,3)
        out1 = out1.view(out1.size(0), -1)  # (?, 1152)
        out2 = self.layer2(x)  # (?,128,3,3)
        out2 = out2.view(out2.size(0), -1)  # (?, 1152)

        cat1 = torch.cat((out1, out2), dim=1)  # (?,1152)
        norm1 = self.lrn(cat1)

        out3, _ = self.inception(x)
        norm2 = self.lrn(out3)

        cat2 = torch.cat((norm1, norm2), dim=1)

        fc1 = self.fc1(cat2)
        out = self.lrn(fc1)
        return out


class Tripletnet(nn.Module):
    def __init__(self, embeddingnet, margin = 1):
        super(Tripletnet, self).__init__()
        self.embeddingnet = embeddingnet
        self.margin = margin

    def forward(self, p, q, n):
        embedded_p = self.embeddingnet(p)
        embedded_q = self.embeddingnet(q)
        embedded_n = self.embeddingnet(n)
        loss = F.triplet_margin_loss(embedded_q,embedded_p,embedded_n, margin = self.margin)
        return loss
