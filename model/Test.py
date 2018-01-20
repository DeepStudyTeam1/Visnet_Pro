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

transform = transforms.Compose ([transforms.Resize ((299, 299)),
                                 transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                                 transforms.ToTensor ()])

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

def test(img_path, vertical, topk = 1000):
    print(img_path)
    img = Image.open(img_path).convert('RGB')
    img = transform(img)
    feature = m1(to_var(img.view(-1,3,299,299)))

    file_path = base_dir + "/feature/" + vertical + ".pkl"
    if torch.cuda.is_available():
        feature_all = torch.load(file_path).cuda()
    else:
        feature_all = torch.load(file_path, map_location=lambda storage, loc: storage)
    feature_all = feature_all - feature.data
    feature_all = feature_all.norm(2,1)
    _ , top_index = torch.topk(feature_all, topk , largest= False)
    with open(base_dir + "/feature/" + vertical + "_feature_id.pkl", 'rb') as f:
        id_list = pickle.load(f)
    top_id = []
    for index in top_index:
        top_id.append(id_list[index])
    return top_id



def eval(vertical):
    with open(base_dir + "/image_lists/" + vertical + "_train.pkl", 'rb') as f:
        q_to_p_map = pickle.load(f)
    sum = 0
    count2 = 0
    for q in q_to_p_map:
        if q_to_p_map[q] == "None":
            continue
        img_path = base_dir + "/images/crop_" + str(q) + ".jpg"
        top_id = test(img_path,vertical,100)
        count = 0
        for p in q_to_p_map[q]:
            if p in top_id:
                count += 1
        acc = count/len(q_to_p_map[q])
        print(acc)
        sum += acc
        count2 += 1
    acc_avg = sum / count2
    print("final acc : %f" %acc_avg)

eval("tops")
