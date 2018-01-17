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
    for item in item_list:
        item_id = item[0]
        path = item_dir + "/" + str(item_id) + ".jpg"
        try:
            Image.open(path)
            count1 += 1
        except:
            print("path doesnt exist %s" %path)
            count2 += 1
    print(count1)
    print(count2)




