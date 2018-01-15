from model import Visnet_Pro
from loader import SingleImage
from torchvision import transforms
from torch.autograd import Variable
import os
import torch
import glob

def to_var(x):
    """Convert tensor to variable."""
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

base_dir = os.path.split(os.getcwd())[0] + "/data/street2shop"
batch_size = 10

m1 = Visnet_Pro()
if torch.cuda.is_available():
    m1 = m1.cuda()
    params = torch.load (base_dir + '/params_skirts_0.pkl')
else:
    params = torch.load (base_dir + '/params_skirts_0.pkl', map_location=lambda storage, loc: storage)
m1.load_state_dict(params)

transform = transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor()])

all_file_paths = glob.glob(base_dir + "/image_lists/*_retrieval.pkl")

file_path = base_dir + "/feature"
if not os.path.exists(file_path):
    os.mkdir(file_path)

def feature(vertical):
    path = base_dir + "/image_lists/" + vertical + "_retrieval.pkl"
    dataset = SingleImage (base_dir + "/images", path, transform=transform)
    loader = torch.utils.data.DataLoader (dataset, batch_size=batch_size, shuffle=True)

    output = []

    for batch_idx, data in enumerate (loader):
        output.append (m1 (to_var (data)))
        if batch_idx % 10 == 0:
            print("Making features [%d/%d]" %(batch_idx, len(loader)))

    torch.save (output, file_path + "/" + vertical + ".pkl")

feature("tops")


