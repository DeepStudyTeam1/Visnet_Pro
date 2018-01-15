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

batch_size = 100
epochs = 1
learning_rate = 0.01

# create network
m1 = Visnet_Pro ()
m2 = Tripletnet (m1)
if torch.cuda.is_available ():
    m1 = m1.cuda ()
    m2 = m2.cuda ()

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

        print ('Epoch [%d/%d], Iter [%d/%d],  Loss: %.4f'
               % (i, epochs, batch_idx, len (dataset) // batch_size, loss.data))
        if batch_idx % 500 == 0:
            torch.save (m1.state_dict (),
                        base_dir + '/params_' + str (i) + "_" + str (batch_idx) + '.pkl')
torch.save (m1.state_dict (),base_dir + '/params_final.pkl')
print ("train success")
