import os
import pickle
import workerpool
from urllib.parse import urlparse
import requests
from PIL import Image
from io import BytesIO
import traceback

__author__ = "ananya.h"
base_dir = "C:/Users/cksdn/Downloads/visnet_mine/data/street2shop"


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
                r = requests.get (urlObj.url, timeout = 3)
                if r.status_code == 200:
                    i = Image.open (BytesIO (r.content))
                    i.save (self.destination_path + "/" + urlObj.id + ".jpg")
                else:
                    return [r.status_code, urlObj]
            except:
                print("Download failed!!!")
                print(urlObj.url.split (".")[-1][:3])
                print(urlObj.id)
                if os.path.exists(self.destination_path + "/" + urlObj.id + ".jpg"):
                    os.remove(self.destination_path + "/" + urlObj.id + ".jpg")
                # traceback.print_exc()
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
    url_file_path = base_dir + "/photos/photos.txt"
    dst_dir = base_dir + "/images"
    images_downloaded = os.listdir (dst_dir)
    ids_downloaded = set ([x.split (".")[0] for x in images_downloaded])
    with open (url_file_path, 'r') as urlFile:
        lines = urlFile.readlines ()
        lines = [x.strip () for x in lines]
        lines = [x.split (",")[:2] for x in lines]
    url_objects = {}
    for line in lines:
        img_id, url = line
        img_id = str (int (img_id))
        if img_id not in ids_downloaded:
            url_objects[img_id] = URLObject (img_id, url)  # Done to remove duplicates
    url_objects = url_objects.values ()
    print ("Commencing downloads for " + str (len (url_objects)) + " urls")
    downloader = ParallelImageDownloader (25, dst_dir)
    errors = downloader.download_batch (url_objects)
    with open (base_dir + "/tmp/errors.pkl", "wb") as pklFile:
        pickle.dump (errors, pklFile)