import os
from model import Visnet_Pro
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
import torch
import numpy as np
import pickle
import requests
from io import BytesIO

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
    params = torch.load (base_dir + '/params_final.pkl')
else:
    params = torch.load (base_dir + '/params_final.pkl', map_location=lambda storage, loc: storage)

m1.load_state_dict(params)
m1.eval()

def download_one_image (id):
    if os.path.exists(base_dir + "/images/" + str(id) + ".jpg"):
        print("this file is already exist!!")
        return True
    with open (base_dir + "/photos/photos.txt", 'r') as urlFile:
        line = urlFile.readlines ()[id - 1]
        line = line.strip ()
        line = line.split (",")
        print(line[0])
        url = line[-1]
        print (url)
        try:
            r = requests.get (url, timeout = 10)
            i = Image.open (BytesIO (r.content))
            i.save (base_dir + "/images/" + str (id) + ".jpg")
            print("Download Success!!!")
            return True
        except:
            print ("Download failed!!!")
            if os.path.exists (base_dir + "/images" + str(id) + ".jpg"):
                os.remove (base_dir + "/images" + str(id) + ".jpg")
            return False

def show_image(predictions):
    for i in predictions:
        download_one_image(i)
        path = base_dir + "/images/" + str(i) + ".jpg"
        Image.open(path).show()

def test(img_id, vertical, topk = 100):

    print(img_id)
    show_image([img_id])
    img_path = base_dir + "/images/"+ str(img_id) + ".jpg"

    img = Image.open(img_path).convert('RGB')
    img = transform(img)
    feature = m1(to_var(img.view(-1,3,299,299)))

    file_path = base_dir + "/feature/" + vertical + ".pkl"
    if torch.cuda.is_available():
        feature_all = torch.load(file_path)
    else:
        feature_all = torch.load(file_path, map_location=lambda storage, loc: storage)
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
    return top_id

vertical = "dresses"
with open(base_dir + "/image_lists/" + vertical + "_test.pkl", 'rb') as f:
    q_to_p_map = pickle.load(f)


print(q_to_p_map)
query_list = list(q_to_p_map.keys())
img_id = query_list[2]
top_id = test(57591, vertical)
count = 0

for i in top_id:
    if i in q_to_p_map[img_id]:
        count += 1.0

print("Acc : %.3f" % (count / len(q_to_p_map)))

show_image(top_id)