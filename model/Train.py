from loader import TripletImage
from model import Visnet_Pro, Tripletnet
import os
import torch
from torchvision import transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn


def to_var (x):
    """Convert tensor to variable."""
    if torch.cuda.is_available ():
        x = x.cuda ()
    return Variable (x)


verticals = ["tops", "dresses", "outerwear", "skirts"]

# parameter
base_dir = os.path.split (os.getcwd ())[0] + "/data/street2shop"

batch_size = 10
epochs = 1
learning_rate = 0.01

# create network
m1 = Visnet_Pro (heavy = True)
m2 = Tripletnet (m1)
if torch.cuda.is_available ():
    m1 = m1.cuda ()
    m2 = m2.cuda ()
    params = torch.load (base_dir + '/params_final_28.pkl')
else:
    params = torch.load (base_dir + '/params_final_28.pkl', map_location=lambda storage, loc: storage)
# m1.load_state_dict(params)


grad_param = []
for param in m2.parameters ():
    if param.requires_grad == True:
        grad_param.append (param)

cudnn.benchmark = True

transform = transforms.Compose ([transforms.Resize ((299, 299)), transforms.ToTensor ()])

optimizer = torch.optim.Adam (grad_param, lr=learning_rate)

dataset = TripletImage (image_path=base_dir + "/images",
                        verticals=verticals,
                        transform=transform)
train_loader = torch.utils.data.DataLoader (dataset, batch_size=batch_size, shuffle=True)

# train start
for i in range (epochs):
    for batch_idx, (data1, data2, data3) in enumerate (train_loader):
        data1 = to_var (data1)
        data2 = to_var (data2)
        data3 = to_var (data3)

        loss = m2 (data1, data2, data3)

        optimizer.zero_grad ()
        loss.backward ()
        optimizer.step ()

        print ('Epoch [%d/%d], Iter [%d/%d],  Loss: %.8f'
               % (i, epochs, batch_idx, len (dataset) // batch_size, loss.data))
        if batch_idx % 1000 == 0:
            torch.save (m1.state_dict (),
                        base_dir + '/params_final_heavy_' + str (batch_idx) + '.pkl')
torch.save (m1.state_dict (),base_dir + '/params_final_heavy.pkl')
print ("train success")
