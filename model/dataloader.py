import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


my_csv_path = "triplets.csv"


def get_id_batch (file_path, batch_size=5):
    with open (file_path, 'r') as file:
        lines = file.readlines()
        batch_num = int(len(lines)/ (batch_size))
        lines = lines[:batch_size*batch_num]
        lines = [x.split(',')[:-1] for x in lines]
        for i in range(0,batch_num*batch_size,batch_size):
            yield lines[i:i+batch_size]

def get_image(id_list):
    image_p=[]
    image_q=[]
    image_n=[]
    for id_triplet in id_list:
        img = Image.open("C:/Users/cksdn/Downloads/visnet_mine/data/street2shop/images/"+str(id_triplet[0])+ ".jpg")
        img = img.resize((299,299))
        img = np.array(img, dtype=np.float32)
        image_q.append(img)
        img = Image.open (
            "C:/Users/cksdn/Downloads/visnet_mine/data/street2shop/images/" + str (id_triplet[1]) + ".jpg")
        img = img.resize ((299, 299))
        img = np.array (img, dtype=np.float32)
        image_p.append (img)
        img = Image.open (
            "C:/Users/cksdn/Downloads/visnet_mine/data/street2shop/images/" + str (id_triplet[2]) + ".jpg")
        img = img.resize ((299, 299))
        img = np.array (img, dtype=np.float32)
        image_n.append (img)
    out = [image_q, image_p, image_n]
    return out



#for _, x in enumerate(get_id_batch(my_csv_path)):
#    print(    len(       get_image(x)[0]     )          )