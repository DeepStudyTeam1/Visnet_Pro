import os
import glob
import json
import pickle
import workerpool
from urllib.parse import urlparse
import requests
from PIL import Image
from io import BytesIO
import traceback
import random
import csv

base_dir = "C:/Users/cksdn/Documents/GitHub/Visnet_Pro/data/street2shop"
meta_dir = os.path.join (base_dir, "meta", "json")
img_dir = os.path.join (base_dir, "images")
structured_dir = os.path.join (base_dir, "structured_images")
url_file_path = base_dir + "/photos/photos.txt"
failed_lists = set()

def download_one_image (id):
    if id in failed_lists:
        print("this file is already failed!!")
        return False
    if os.path.exists(img_dir + "/" + str(id) + ".jpg"):
        print("this file is already exist!!")
        return True
    with open (url_file_path, 'r') as urlFile:
        line = urlFile.readlines ()[id - 1]
        line = line.strip ()
        line = line.split (",")
        url = line[-1]
        print (url)
        try:
            r = requests.get (url, timeout = 10)
            i = Image.open (BytesIO (r.content))
            i.save (img_dir + "/" + str (id) + ".jpg")
            print("Download Success!!!")
            return True
        except:
            print ("Download failed!!!")
            failed_lists.add(id)
            if os.path.exists (img_dir + "/" + str(id) + ".jpg"):
                os.remove (img_dir + "/" + str(id) + ".jpg")
            return False


def triplet_make (vertical, num1=5, num2=5, train=True, crop=True):
    prefix = "train" if train else "test"
    filename = prefix + "_pairs_" + vertical + ".json"
    retrieval_path = os.path.join (meta_dir, "retrieval_" + vertical + ".json")
    # query_dir = os.path.join (structured_dir, "wtbi_" + vertical + "_query_crop")
    output_path = os.path.join(base_dir, "triplet", vertical)
    with open (os.path.join (meta_dir, filename)) as jsonFile:
        pairs = json.load (jsonFile)
    photo_to_product_map = {}
    with open (retrieval_path) as jsonFile:
        data = json.load (jsonFile)
    print(len(data))
    for info in data:
        photo_to_product_map[info["photo"]] = info["product"]
    product_to_photo_map = {}
    for photo in photo_to_product_map:
        product = photo_to_product_map[photo]
        if product not in product_to_photo_map:
            product_to_photo_map[product] = set ()
        product_to_photo_map[product].add (photo)
    count = 0
    for pair in pairs[:num1]:
        if not download_one_image (pair["photo"]):
            print ("no query!")
            continue
        photo = pair["photo"]
        product = pair["product"]
        p_s = []
        for i in product_to_photo_map[product]:
            if not download_one_image (i):
                print ("no pos")
                continue
            p_s.append (i)
        triplets = []
        for p in p_s:
            for j in range (num2):
                q_id = str (photo)
                p_id = str (p)
                n_index = random.randint (0, len (data) - 1)
                if not download_one_image (data[n_index]["photo"]):
                    print ("no neg!")
                    continue
                n = data[n_index]["photo"]
                if n not in p_s and n != photo:
                    n_id = str (n)
                    triplets.append ([q_id, p_id, n_id, vertical])
                    print ("yes triplet")
            if not os.path.exists (output_path):
                os.mkdir (output_path)
            with open (output_path + "/triplets.csv", "a+") as csvFile:
                triplets = [[x[0], x[1], x[2], x[3]] for x in triplets]
                for triplet in triplets:
                    triplet = ",".join(triplet) + "\n"
                    csvFile.write (triplet)
                count += len(triplets)
                triplets = []
    print(count)


if __name__ == "__main__":
    triplet_make ("dresses", num1 = 5, num2 = 5 )
