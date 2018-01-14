import glob
import json
import random
import csv
import os
import pickle


base_dir = os.path.split(os.getcwd())[0] + "/data/street2shop"

def sample(verticals):
    file_dir = os.path.join(base_dir, "image_lists")
    triplet_dir = os.path.join(base_dir, "triplet")
    if not os.path.exists(triplet_dir):
        os.mkdir(triplet_dir)
    number_of_n = 3
    for vertical in verticals:
        with open(os.path.join(file_dir, vertical + "_retrieval.pkl"), 'rb') as f:
            universe = pickle.load(f)
        with open(os.path.join(file_dir, vertical + "_train.pkl"), 'rb') as f:
            q_to_p_map = pickle.load(f)
        triplets = []
        for query in q_to_p_map:
            if q_to_p_map[query] == "None":
                continue
            for pos in q_to_p_map[query]:
                temp = [query]
                temp.append(pos)
                for i in range(number_of_n):
                    neg = random.randint(0, len(universe)-1)
                    neg = universe[neg]
                    triplet = list(temp)
                    if neg not in q_to_p_map[query]:
                        triplet.append(neg)
                        triplets.append(triplet)
        with open(triplet_dir + "/" + vertical + ".pkl", "wb") as f:
            pickle.dump(triplets, f)
        print("Complete making triplets of " + vertical + " " + str(len(triplets)))




if __name__ == "__main__":
    all_file_paths = glob.glob(base_dir + "/meta/json/retrieval_*.json")
    verticals = []
    for path in all_file_paths:
        vertical = os.path.split(path)[-1].split(".")[0].split("_")[-1]
        verticals.append(vertical)
    print(verticals)
    sample(verticals)
