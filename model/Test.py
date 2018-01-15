import os
from model import Visnet_Pro
from PIL import Image
from torchvision import transforms
import torch
from sklearn.neighbors import NearestNeighbors
import pickle

base_dir = os.path.split(os.getcwd())[0] + "/data/street2shop"

transform = transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor()])

m1 = Visnet_Pro()
m1.load_state_dict(torch.load('params.pkl'))

def show_image(predictions):
    for i in predictions:
        path = base_dir + "/images/" + str(i) + ".jpg"
        Image.open(path).show()

def test(img_id, vertical):

    show_image([img_id])

    img_path = base_dir + "/images/"+ str(img_id) + ".jpg"

    img = Image.open(img_path).convert('RGB')
    img = transform(img)
    feature = m1(img)

    file_path = base_dir + "/feature/" + vertical + ".pkl"
    feature_all = torch.load(file_path)

    knn = NearestNeighbors(n_neighbors=10)
    knn.fit(feature_all)

    predict = knn.kneighbors(feature, return_distance=False)
    show_image(predict)

    with open(base_dir + "/image_lists/" + vertical + "_test.pkl", 'rb') as f:
        q_to_p_map = pickle.load(f)

    count = 0

    for i in predict:
        if i in q_to_p_map[img_id]:
            count += 1.0

    print("Acc : %.3f" %(count / len(q_to_p_map)))