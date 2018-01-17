import os
import pickle
import workerpool
from urllib.parse import urlparse
import requests
from PIL import Image
from io import BytesIO
import os
import traceback
import glob



base_dir = os.path.split(os.getcwd())[0] + "/data/looky"

class URLObject (object):
    def __init__ (self, id, url):
        self.id = str (id)
        self.url = url

class ParallelImageDownloader (object):
    def __init__ (self, max_pool_size, destination_path):
        self.max_pool_size = max_pool_size
        self.destination_path = destination_path

    def is_url (self, url):
        return url is not None and urlparse (url).scheme != ""

    def download_image (self, urlObj):
        if self.is_url (urlObj.url):
            try:
                r = requests.get (urlObj.url, timeout = 5)
                if r.status_code == 200:
                    i = Image.open (BytesIO (r.content))
                    i.save (self.destination_path + "/" + urlObj.id + ".jpg")
                else:
                    return [r.status_code, urlObj]
            except:
                print("Download failed!!!")
                print(urlObj.url)
                print(urlObj.url.split (".")[-1][:3])
                print(urlObj.id)
                if os.path.exists(self.destination_path + "/" + urlObj.id + ".jpg"):
                    os.remove(self.destination_path + "/" + urlObj.id + ".jpg")
                return [-1, urlObj]
        else:
            return None

    def download_batch (self, urlObjects):
        pool = workerpool.WorkerPool (min (self.max_pool_size, len (urlObjects)))
        errors = pool.map (self.download_image, urlObjects)
        pool.shutdown ()
        pool.wait ()
        errors = list(filter ((lambda x: x), errors))
        print ("Number of images sent for download " + str (len (urlObjects)))
        print ("Number of images that failed " + str (len (errors)))
        return errors


if __name__ == "__main__":
    item_dir = base_dir + "/item"
    all_item_path = glob.glob(item_dir + "/*.pkl")
    image_dir = base_dir + "/images"
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)
    images_downloaded = os.listdir (image_dir)
    ids_downloaded = set ([x.split (".")[0] for x in images_downloaded])

    for item_path in all_item_path:
        with open(item_path, 'rb') as f:
            lines = pickle.load(f)
        url_objects = {}
        for line in lines:
            img_id = line[0]
            id_for_url = "%09d" %int(img_id)
            url = "https://s3.ap-northeast-2.amazonaws.com/looky/" + id_for_url + ".jpg"
            if img_id not in ids_downloaded:
                url_objects[img_id] = URLObject(img_id, url)  # Done to remove duplicates
        url_objects = url_objects.values()
        print("Commencing downloads for " + str(len(url_objects)) + " urls")
        downloader = ParallelImageDownloader(25, image_dir)
        _ = downloader.download_batch(url_objects)
