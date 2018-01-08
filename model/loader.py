from PIL import Image
import os
import os.path

import torch.utils.data
import torchvision.transforms as transforms

def default_image_loader(path):
    return Image.open(path).convert('RGB')

class TripletImageLoader(torch.utils.data.Dataset):
    def __init__(self, base_path, filenames_filename, triplets_file_name, transform=None,
                 loader=default_image_loader):
        """ filenames_filename: A text file with each line containing the path to an image e.g.,
                images/class1/sample.jpg
            triplets_file_name: A text file with each line containing three integers,
                where integer i refers to the i-th image in the filenames file.
                For a line of intergers 'a b c', a triplet is defined such that image a is more
                similar to image c than it is to image b, e.g.,
                0 2017 42 """
        self.base_path = base_path      #/visnet_pro/data/street2shop/images
        triplets = []
        for line in open(triplets_file_name):
            line = line.split(",")
            line = [str(x)+".jpg" for x in line]
            triplets.append(line[0:3])
        self.triplets = triplets
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img1 = self.loader(os.path.join(self.base_path,self.triplets[index][0]))
        img2 = self.loader(os.path.join(self.base_path,self.triplets[index][1]))
        img3 = self.loader(os.path.join(self.base_path,self.triplets[index][2]))
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        return img1, img2, img3

    def __len__(self):
        return len(self.triplets)