from PIL import Image
import os
import glob

base_dir = os.path.split(os.getcwd())[0] + "/data/street2shop"

all_image_path = glob.glob(base_dir + "/images/*.jpg")
count = 0
for image_path in all_image_path:
    try:
        Image.open(image_path)
    except:
        count = count + 1
        print(image_path)
        os.remove(image_path)
print(count)
