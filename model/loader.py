from PIL import Image
import os
import os.path
import torch.utils.data
import pickle


def default_image_loader(path):
    return Image.open(path).convert('RGB')

base_dir = os.path.split(os.getcwd())[0] + "/data/street2shop"

class TripletImage(torch.utils.data.Dataset):
    def __init__(self, image_path, verticals, transform=None,
                 loader=default_image_loader):
        self.image_path = image_path
        self.triplets = []
        for vertical in verticals:
            triplets_file_path = base_dir + "/triplet/" + vertical + ".pkl"
            with open (triplets_file_path, 'rb') as f:
                self.triplets.extend(pickle.load (f))
        self.transform = transform
        self.loader = loader
        print("Load dataset! length: " + str(len(self.triplets)))

    def __getitem__(self, index):
        img1 = self.loader(self.image_path + "/crop_" + str(self.triplets[index][0]) + ".jpg")
        img2 = self.loader(self.image_path + "/" + str(self.triplets[index][1]) + ".jpg")
        img3 = self.loader(self.image_path + "/" + str(self.triplets[index][2]) + ".jpg")
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return img1, img2, img3

    def __len__(self):
        return len(self.triplets)


class SingleImage(torch.utils.data.Dataset):
    def __init__(self, image_path, single_file_path, transform=None,
                 loader=default_image_loader):
        self.image_path = image_path
        with open(single_file_path, 'rb') as f:
            self.singles = pickle.load(f)
        self.transform = transform
        self.loader = loader
        print("Load dataset! length: " + str(len(self.singles)))

    def __getitem__(self, index):
        img_id = self.singles[index]
        img = self.loader(self.image_path + "/" + str(img_id) + ".jpg")
        if self.transform is not None:
            img = self.transform(img)
        return img_id, img

    def __len__(self):
        return len(self.singles)

class Images(torch.utils.data.Dataset):
    def __init__(self, image_path, transform=None,
                 loader=default_image_loader):
        self.base_path = image_path
        images = os.path.listdir(image_path)
        self.transform = transform
        self.loader = loader
        print("Load dataset! length: " + str(len(images)))

    def __getitem__(self, index):
        img = self.loader(os.path.join(self.base_path, self.images[index]))
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.images)

class SingleImage_for_looky(torch.utils.data.Dataset):
    def __init__(self, image_path, single_file_path, transform=None,
                 loader=default_image_loader):
        self.image_path = image_path
        with open(single_file_path, 'rb') as f:
            self.singles = pickle.load(f)
        self.transform = transform
        self.loader = loader
        print("Load dataset! length: " + str(len(self.singles)))

    def __getitem__(self, index):
        img_id = self.singles[index][0]
        img = self.loader(self.image_path + "/" + str(img_id) + ".jpg")
        if self.transform is not None:
            img = self.transform(img)
        return [img_id, img]

    def __len__(self):
        return len(self.singles)
