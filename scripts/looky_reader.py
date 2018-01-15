import os

base_dir = os.path.split (os.getcwd ())[0] + "/data/looky"
item_path = base_dir + "/item.csv"
image_dir = base_dir + "/images"

if os.path.exists (image_dir):
    os.mkdir (image_dir)

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
         'pants': {"pants", 'COTTON PANTS', 'PANTS', '바지', '팬츠', '청바지', '기모팬츠', '하의', 'Pants'},
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
