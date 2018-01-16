import os
from model import Visnet_Pro
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
import torch
import numpy as np
import pickle


def to_var(x):
    """Convert tensor to variable."""
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

base_dir = os.path.split(os.getcwd())[0] + "/data/street2shop"

transform = transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor()])

m1 = Visnet_Pro ()

if torch.cuda.is_available ():
    m1 = m1.cuda ()
    params = torch.load (base_dir + '/params_0_500.pkl')
else:
    params = torch.load (base_dir + '/params_0_500.pkl', map_location=lambda storage, loc: storage)

m1.load_state_dict(params)
m1.eval()

def show_image(predictions):
    for i in predictions:
        path = base_dir + "/images/" + str(i) + ".jpg"
        Image.open(path).show()

def test(img_id, vertical, topk = 10):

    show_image([img_id])

    img_path = base_dir + "/images/"+ str(img_id) + ".jpg"

    img = Image.open(img_path).convert('RGB')
    img = transform(img)
    feature = m1(to_var(img.view(-1,3,299,299)))

    file_path = base_dir + "/feature/" + vertical + ".pkl"
    feature_all = torch.load(file_path)
    feature_all = feature_all - feature.data
    feature_all = feature_all.norm(2,1)
    _ , top_index = torch.topk(feature_all, topk , largest= False)
    with open(base_dir + "/image_lists/" + vertical + "_retrieval.pkl", 'rb') as f:
        line = pickle.load(f)
    top_id = []
    for index in top_index:
        top_id.append(line[index])

    print("show predict")
    print(top_id)

    show_image(top_id)

    with open(base_dir + "/image_lists/" + vertical + "_test.pkl", 'rb') as f:
        q_to_p_map = pickle.load(f)

    count = 0

    for i in top_id:
        if i in q_to_p_map[img_id]:
            count += 1.0

    print("Acc : %.3f" %(count / len(q_to_p_map)))

test(2337,"tops")