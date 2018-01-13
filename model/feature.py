from model import Visnet_Pro
from loader import SingleImageLoader
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
m1.load_state_dict(torch.load(base_dir + '/params.pkl'))

transform = transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor()])

all_pair_file_paths = glob.glob(base_dir + "/image_lists/*_retrieval.txt")

file_path = base_dir + "/feature"
if not os.path.exists(file_path):
    os.mkdir(file_path)

path = base_dir + "/image_lists/belts_train.txt"
vertical = path.split("_")[0]
dataset = SingleImageLoader(base_dir + "/images", path, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


output = []

for batch_idx, data in enumerate(loader):
    print(data.size())
    output.append (m1(to_var(data)))

torch.save(output, file_path+ "/" + vertical + ".pkl")


