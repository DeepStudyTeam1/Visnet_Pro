import torch
import torch.nn as nn
import torchvision
import os.path
import torch.utils.data
from torch.autograd import Variable
from torchvision import transforms
import pickle

base_dir = os.path.split(os.getcwd())[0] + "/data"

class LRN(nn.Module):
    def __init__ (self):
        super (LRN, self).__init__ ()
    def forward (self, x):
        div = x.norm(2,1, keepdim = True)
        div = div.expand(x.size())
        x = x/div
        return x

class Visnet_Pro(nn.Module):
    def __init__(self, heavy = False):
        print("Create Visnet_Pro!")
        super(Visnet_Pro, self).__init__()
        if heavy == False:
            self.layer1 = nn.Sequential (
                nn.MaxPool2d (kernel_size=5, stride=5, padding=1),  # (?,3,60,60)
                nn.Conv2d (3, 64, kernel_size=6, stride=4, padding=1),  # (?,64,15,15)
                nn.ReLU (),
                nn.BatchNorm2d (64),
                nn.MaxPool2d (kernel_size=7, stride=4, padding=0),  # (?,64,3,3)
            )
            self.layer2 = nn.Sequential (
                nn.MaxPool2d (kernel_size=10, stride=10, padding=1),  # (?,3,30,30)
                nn.Conv2d (3, 64, kernel_size=6, stride=4, padding=0),  # (?,64,7,7)
                nn.ReLU (),
                nn.BatchNorm2d (64),
                nn.MaxPool2d (kernel_size=3, stride=2, padding=0),  # (?,64,3,3)
            )
            self.inception = torchvision.models.inception_v3 (pretrained=True)
            self.inception.training = False
            for param in self.inception.parameters ():
                param.requires_grad = False

            self.inception.fc = nn.Linear (2048, 1024)
            self.fc1 = nn.Linear (2176, 1024)
        else:
            self.layer1 = nn.Sequential (
                nn.MaxPool2d (kernel_size=5, stride=5, padding=1),  # (?,3,60,60)
                nn.Conv2d (3, 256, kernel_size=6, stride=4, padding=1),  # (?,256,15,15)
                nn.ReLU (),
                nn.BatchNorm2d (256),
                nn.MaxPool2d (kernel_size=7, stride=4, padding=0),  # (?,256,3,3)
            )
            self.layer2 = nn.Sequential (
                nn.MaxPool2d (kernel_size=10, stride=10, padding=1),  # (?,3,30,30)
                nn.Conv2d (3, 256, kernel_size=6, stride=4, padding=0),  # (?,256,7,7)
                nn.ReLU (),
                nn.BatchNorm2d (256),
                nn.MaxPool2d (kernel_size=3, stride=2, padding=0),  # (?,256,3,3)
            )
            self.inception = torchvision.models.inception_v3 (pretrained=True)
            self.inception.training = False
            for param in self.inception.parameters ():
                param.requires_grad = False

            self.inception.fc = nn.Linear (2048, 4096)
            self.fc1 = nn.Linear (8704, 4096)

        self.lrn = LRN()

    def forward(self, x):
        out1 = self.layer1(x)  # (?, 128,3,3)
        out1 = out1.view(out1.size(0), -1)  # (?, 1152)
        out2 = self.layer2(x)  # (?,128,3,3)
        out2 = out2.view(out2.size(0), -1)  # (?, 1152)

        cat1 = torch.cat((out1, out2), dim=1)  # (?,1152)
        norm1 = self.lrn(cat1)

        out3 = self.inception(x)
        norm2 = self.lrn(out3)

        cat2 = torch.cat((norm1, norm2), dim=1)

        fc1 = self.fc1(cat2)
        out = self.lrn(fc1)

        return out

def to_var(x):
    """Convert tensor to variable."""
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

model = Visnet_Pro(heavy=True)

if torch.cuda.is_available():
    model = model.cuda()
    params = torch.load(base_dir + '/params_final_100.pkl')
else:
    params = torch.load(base_dir + '/params_final_100.pkl', map_location=lambda storage, loc: storage)

model.load_state_dict(params)
model.eval()

# category : "outer", "shirts", "top", "skirt", "dress"

def find_image(img, category, topk = 20):

    transform = transforms.Compose([transforms.Resize((299, 299)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                    ])
    img = transform(img)

    feature = model(to_var(img.view(-1, 3, 299, 299)))
    file_path = base_dir + "/data/" + category + "_feature.pkl"

    if torch.cuda.is_available():
        feature_id, feature_all = torch.load(file_path)
        feature_all = feature_all.cuda()
    else:
        feature_id, feature_all = torch.load(file_path, map_location=lambda storage, loc: storage)

    feature_all = feature_all - feature.data
    feature_all = feature_all.norm(2, 1)
    _, top_index = torch.topk(feature_all, topk, largest=False)

    top_id = []
    for index in top_index:
        top_id.append(feature_id[index])

    with open(base_dir + "/feature/" + category + ".pkl", 'rb') as f:
        item_list = pickle.load(f)

    return_list = []
    for id in top_id:
        for item in item_list:
            if item[0] == id:
                return_list.append([item[1], item[2], item[3]])  # url, name, price
                break

    return return_list




