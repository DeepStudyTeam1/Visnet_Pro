import os
import requests
from io import BytesIO
from PIL import Image

base_dir = "C:/Users/cksdn/Downloads/fk-visual-search-master/data/street2shop"
url_file_path = base_dir + "/photo/photos.txt"
destination_path = base_dir + "/images"
id = 380

with open (url_file_path, 'r') as urlFile:
    line = urlFile.readlines ()[id - 1]
    line = line.strip ()
    line = line.split (",")
    url = line[-1]
    print(url)
    print(str(id))
    try:
        r = requests.get (url)
        print("1")
        i = Image.open (BytesIO (r.content))
        print("2")
        i.save (destination_path + "/" + str (id) + ".jpg")
    except:
        print ("Download failed!!!")
        print (url.split (".")[-1][:3])
        print (id)


