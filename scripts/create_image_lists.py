import glob
import json
import os

base_dir = os.path.split(os.getcwd())[0] + "/data/street2shop"
meta_dir = os.path.join(base_dir, "meta", "json")
image_dir = os.path.join(base_dir, "images")
structured_dir = os.path.join(base_dir, "image_lists")
if not os.path.exists(structured_dir):
    os.mkdir(structured_dir)
all_pair_file_paths = glob.glob(meta_dir + "/retrieval_*.json")

for path in all_pair_file_paths:

    vertical = path.split("_")[-1].split(".")[0]
    retrieval_path = os.path.join(structured_dir, vertical + "_retrieval.txt")
    train_path = os.path.join(structured_dir, vertical + "_train.txt")
    test_path = os.path.join(structured_dir, vertical + "_test.txt")

    photo_to_product_map = {}
    with open(path) as jsonFile:
        data = json.load(jsonFile)
    for info in data:
        img_path = os.path.join(image_dir, str(info["photo"]) + ".jpg")
        if os.path.exists(img_path):
            photo_to_product_map[info["photo"]] = info["product"]

    product_to_photo_map = {}
    for photo in photo_to_product_map:
        product = photo_to_product_map[photo]
        if product not in product_to_photo_map:
            product_to_photo_map[product] = set()
        product_to_photo_map[product].add(photo)

    print("Create retrieval ids list for %s" % vertical)

    count = 0
    with open(retrieval_path, "w") as f:
        for photo_id in photo_to_product_map:
            f.write(str(photo_id) + "," + str(photo_to_product_map[photo_id]) + "\n")
            count += 1
    print("retrieval count: " + str(count))

    train_file = "train_pairs_" + vertical + ".json"
    train_photo_to_product_map = {}
    with open(os.path.join(meta_dir, train_file)) as jsonFile:
        data = json.load(jsonFile)
    for info in data:
        img_path = os.path.join(image_dir, str(info["photo"]) + ".jpg")
        if os.path.exists(img_path):
            train_photo_to_product_map[info["photo"]] = info["product"]

    print("Create train ids list for %s" % vertical)

    count = 0
    with open(train_path, "w") as f:
        for photo_id in train_photo_to_product_map:
            product_id = train_photo_to_product_map[photo_id]
            try:
                f.write(str(photo_id) + "," + str(product_id) + "," + str(product_to_photo_map[product_id]) + "\n")
            except:
                f.write(str(photo_id) + "," + str(product_id) + "\n")
            count += 1
    print("train count: " + str(count))

    test_file = "test_pairs_" + vertical + ".json"
    test_photo_to_product_map = {}
    with open(os.path.join(meta_dir, test_file)) as jsonFile:
        data = json.load(jsonFile)
    for info in data:
        img_path = os.path.join(image_dir, str(info["photo"]) + ".jpg")
        if os.path.exists(img_path):
            test_photo_to_product_map[info["photo"]] = info["product"]

    print("Create test ids list for %s" % vertical)

    count = 0
    with open(test_path, "w") as f:
        for photo_id in test_photo_to_product_map:
            product_id = test_photo_to_product_map[photo_id]
            try:
                f.write(str(photo_id) + "," + str(product_id) + "," + str(product_to_photo_map[product_id]) + "\n")
            except:
                f.write(str(photo_id) + "," + str(product_id) + "\n")
            count += 1
    print("test count: " + str(count))
