import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Visnet_Pro(nn.Module):
    def __init__(self):
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

        self.bn1 = nn.BatchNorm1d(1152)

        self.inception = torchvision.models.inception_v3(pretrained=True)
        for param in self.inception.parameters():
            param.requires_grad = False

        self.inception.fc = nn.Linear(2048, 1024)
        self.fc1 = nn.Linear(2176, 1024)

        self.bn2 = nn.BatchNorm1d(1024)
        self.bn3 = nn.BatchNorm1d(1024)
    def forward(self, x):
        out1 = self.layer1(x)  # (?, 128,3,3)
        out1 = out1.view(out1.size(0), -1)  # (?, 1152)
        out2 = self.layer2(x)  # (?,128,3,3)
        out2 = out2.view(out2.size(0), -1)  # (?, 1152)

        cat1 = torch.cat((out1, out2), dim=1)  # (?,1152)
        norm1 = self.bn1(cat1)

        if self.training ==True:
            out3, _ = self.inception(x)
        else:
            out3 = self.inception(x)
        norm2 = self.bn2(out3)

        cat2 = torch.cat((norm1, norm2), dim=1)

        fc1 = self.fc1(cat2)
        out = self.bn3(fc1)
        return out


class Tripletnet(nn.Module):
    def __init__(self, embeddingnet, margin = 1):
        super(Tripletnet, self).__init__()
        self.embeddingnet = embeddingnet
        self.margin = margin

    def forward(self, q, p, n):
        embedded_q = self.embeddingnet(q)
        embedded_p = self.embeddingnet(p)
        embedded_n = self.embeddingnet(n)
        loss = F.triplet_margin_loss(embedded_q,embedded_p,embedded_n, margin = self.margin)
        return loss
