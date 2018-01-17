from model import Visnet_Pro
from loader import SingleImage_for_looky
from torchvision import transforms
from torch.autograd import Variable
import os
import torch
import glob

def to_var(x):
    """Convert tensor to variable."""
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile = True)

base_dir = os.path.split(os.getcwd())[0] + "/data/looky"
batch_size = 100

m1 = Visnet_Pro()
if torch.cuda.is_available():
    m1 = m1.cuda()
    params = torch.load (base_dir + '/params_final.pkl')
else:
    params = torch.load (base_dir + '/params_final.pkl', map_location=lambda storage, loc: storage)

m1.load_state_dict(params)
m1.eval()

transform = transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor()])

file_path = base_dir + "/feature"
if not os.path.exists(file_path):
    os.mkdir(file_path)

def feature(verticals):
   for vertical in verticals:
       path = base_dir + "/item/" + vertical + ".pkl"
       dataset = SingleImage_for_looky(base_dir + "/images", path, transform=transform)
       loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

       output = torch.Tensor(0,0)

       if torch.cuda.is_available():
           output = output.cuda()

        id_list = []

       for batch_idx, data in enumerate(loader):
           id, out = data
           out = m1(to_var(out)).data
           output = torch.cat((output, out), 0)
           id_list.extend(id)
           if batch_idx % 10 == 0:
               print("Making features [%d/%d]" % (batch_idx, len(loader)))

       torch.save(output, file_path + "/" + vertical + ".pkl")
       with open(base_dir + "/aigo.txt", 'w') as f:
           f.write(id_list)

feature(["top"])


