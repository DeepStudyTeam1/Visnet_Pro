import os
import pickle

base_dir = os.path.split (os.getcwd ())[0] + "/data/looky"
item_dir = base_dir + "/item"
item_path = item_dir + "/item.csv"

if not os.path.exists(item_dir):
    os.mkdir(item_dir)




with open (item_path, 'r', encoding='UTF8') as f:
    lines = f.readlines ()
line = lines[0]
print (line)
line = line.split (";")
print (line[0])  # id
print (line[1])  # url
print (line[2])  # name
print (line[3])  # price
print (line[4])  # shop
print (line[5])  # big category
print (line[6])  # small category
print (line[9])  # True
print (line[10])  # date
dict1 = {}
for line in lines:
    line = line.replace ('"', "").split (";")
    if line[5] not in dict1.keys ():
        dict1[line[5]] = 1
    else:
        dict1[line[5]] += 1
print (dict1)
dict2 = {'outer': {"outer", 'OUTER', '아우터', '패딩', '코트', 'Outer'},
         'shirts': {"shirts", 'SHIRT', '셔츠', '남방/와이셔츠', 'SHIRTS'},
         'top': {"top", 'TOP', '상의', '티셔츠', '니트', 'TEE', 'KNIT', '탑', 'Top'},
         'skirt': {'skirt', 'SKIRT', '스커트'},
         'dress': {'onepiece', 'DRESS', '드레스', 'dress', '원피스', 'ONEPIECE'}
         }
dict3 = {}
for key1 in dict1:
    for key2 in dict2:
        if key1 in dict2[key2]:
            if key2 in dict3:
                dict3[key2] += dict1[key1]
            else:
                dict3[key2] = dict1[key1]
print(dict3)
new_line = []
new_lines1 = []
new_lines2 = []
new_lines3 = []
new_lines4 = []
new_lines5 = []
for line in lines:
    line = line.replace('"', "").strip("\n").split(";")
    if line[9] == "0":
        continue
    else:
        new_line = [line[0], line[1], line[2], line[3], line[4], line[6], line[10]] # id, url, name, price, shop, small, date
        if line[5] in {"outer", 'OUTER', '아우터', '패딩', '코트', 'Outer'}:
            new_line.append("outer")
            new_lines1.append(new_line)
        elif line[5] in {"shirts", 'SHIRT', '셔츠', '남방/와이셔츠', 'SHIRTS'}:
            new_line.append("shirts")
            new_lines2.append(new_line)
        elif line[5] in {"top", 'TOP', '상의', '티셔츠', '니트', 'TEE', 'KNIT', '탑', 'Top'}:
            new_line.append("top")
            new_lines3.append(new_line)
        elif line[5] in {'skirt', 'SKIRT', '스커트'}:
            new_line.append("skirt")
            new_lines4.append(new_line)
        elif line[5] in {'onepiece', 'DRESS', '드레스', 'dress', '원피스', 'ONEPIECE'}:
            new_line.append("dress")
            new_lines5.append(new_line)
        else:
            continue

with open(item_dir + "/outer.pkl", 'wb') as f:
    pickle.dump(new_lines1, f)
with open(item_dir + "/shirts.pkl", 'wb') as f:
    pickle.dump(new_lines2, f)
with open(item_dir + "/top.pkl", 'wb') as f:
    pickle.dump(new_lines3, f)
with open(item_dir + "/skirt.pkl", 'wb') as f:
    pickle.dump(new_lines4, f)
with open(item_dir + "/dress.pkl", 'wb') as f:
    pickle.dump(new_lines5, f)

with open(item_dir + "/top.pkl", 'rb') as f:
    new = pickle.load(f)
print("!!!")
print(new[0])

dict4 = {}
for line in lines:
    line = line.replace ('"', "").split (";")
    if line[4] not in dict4.keys ():
        dict4[line[4]] = 1
    else:
        dict4[line[4]] += 1
print(dict4)


