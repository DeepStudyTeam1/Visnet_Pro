from PIL import Image
import pickle
import glob
import os

base_dir = os.path.split(os.getcwd())[0] + "/data/looky"
img_dir = base_dir + "/images"
item_dir = base_dir + "/item"

all_item_path = glob.glob(item_dir + "/*.pkl")

for item_path in all_item_path:
    with open(item_path, 'rb') as f:
        item_list = pickle.load(f)
    count1 = 0
    count2 = 0
    count3 = 0
    for item in item_list:
        item_id = item[0]
        path = img_dir + "/" + str(item_id) + ".jpg"
        try:
            Image.open(path)
            count1 += 1
        except:
            if os.path.exists(path):
                os.remove(path)
                count3 += 0
            count2 += 1
    print("success %d" %count1)
    print("fail %d" %count2)
    print("delete %d" %count3)




