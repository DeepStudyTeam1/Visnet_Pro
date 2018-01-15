import glob
import random
import os
import pickle


base_dir = os.path.split(os.getcwd())[0] + "/data/street2shop"

def sample(verticals, number_of_n = 3):
    file_dir = os.path.join(base_dir, "image_lists")
    triplet_dir = os.path.join(base_dir, "triplet")
    if not os.path.exists(triplet_dir):
        os.mkdir(triplet_dir)
    count = 0
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
        count += len(triplets)
        print("Complete making triplets of " + vertical + " " + str(len(triplets)))
    print("Complete making tripkets " + str(count))




if __name__ == "__main__":
    sample(["dresses"],6)
    sample(["outerwear"], 31)
    sample(["tops"], 17)
    sample(["skirts"], 28)
