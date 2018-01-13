from model import Visnet_Pro
from loader import SingleImageLoader
from torchvision import transforms
import os
import torch
import glob

base_dir = os.path.split(os.getcwd())[0] + "/data/street2shop"
batch_size = 100

m1 = Visnet_Pro()
m1.load_state_dict(torch.load('params.pkl'))

transform = transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor()])

all_pair_file_paths = glob.glob(base_dir + "/image_lists/*_retrieval.txt")

# for path in all_pair_file_paths:
path = base_dir + "/image_lists/belts_retrieval.txt"
vertical = path.split("_")[0]
dataset = SingleImageLoader(base_dir + "/images", path, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
for batch_idx, data in enumerate(loader):
    print(data.size())
    output = m1(data)
    f_handle = file(filename, 'a')
    numpy.save(f_handle, arr)
    f_handle.close()


