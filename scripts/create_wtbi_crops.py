import glob
import json
import os
import cv2


base_dir = os.path.split(os.getcwd())[0] + "/data/street2shop"
meta_dir = os.path.join(base_dir, "meta", "json")
images_dir = os.path.join(base_dir, "images")
query_files = glob.glob(meta_dir + "/*_pairs_*.json")

for path in query_files:
    vertical = path.split("_")[-1].split(".")[0]
    print("Processing path %s" %(vertical))
    with open(path) as jsonFile:
        pairs = json.load(jsonFile)
    for pair in pairs:
        try:
            query_id = pair["photo"]
            bbox = pair["bbox"]
            query_path = os.path.join(images_dir, str(query_id) + ".jpg")
            if not os.path.exists(query_path):
                continue
            if os.path.exists(os.path.join(images_dir,  "crop_" + str(query_id) + ".jpg")):
                continue
            img = cv2.imread(query_path, cv2.IMREAD_COLOR)
            x, w, y, h = int(bbox["left"]), int(bbox["width"]), int(bbox["top"]), int(bbox["height"])
            crop_img = img[y:y + h, x:x + w]
            cv2.imwrite(images_dir + "/crop_" + str(query_id) + ".jpg", crop_img)
            print("Crop " + str(query_id) + ".jpg")
        except:
            print("error")
