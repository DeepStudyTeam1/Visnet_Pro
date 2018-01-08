from PIL import Image
import os
import os.path

import torch.utils.data
import torchvision.transforms as transforms


def default_image_loader(path):
    return Image.open(path).convert('RGB')


class TripletImageLoader(torch.utils.data.Dataset):
    def __init__(self, base_path, triplets_file_path, transform=None,
                 loader=default_image_loader):
        self.base_path = base_path  # /visnet_pro/data/street2shop/images
        triplets = []
        for line in open(triplets_file_path):
            line = line.split(",")
            line = [str(x) + ".jpg" for x in line]
            triplets.append(line[0:3])
        self.triplets = triplets
        print(len(triplets))
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img1 = self.loader(os.path.join(self.base_path, self.triplets[index][0]))
        img2 = self.loader(os.path.join(self.base_path, self.triplets[index][1]))
        img3 = self.loader(os.path.join(self.base_path, self.triplets[index][2]))
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return img1, img2, img3

    def __len__(self):
        return len(self.triplets)
