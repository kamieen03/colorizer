import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def default_loader(path):
    return Image.open(path).convert('RGB')


class Dataset(data.Dataset):
    def __init__(self, dataPath, fineSize, test=False):
        super(Dataset,self).__init__()
        self.dataPath = dataPath
        self.image_list = [x for x in os.listdir(dataPath) if is_image_file(x)]
        self.image_list = sorted(self.image_list)

        if not test:
            self.transform = transforms.Compose([
                                 transforms.Resize(fineSize),
                                 transforms.RandomHorizontalFlip()
                             ])
        else:
            self.transform = transforms.Compose([
                                 transforms.Resize(fineSize)
                             ])

        self.test = test

    def __getitem__(self,index):
        dataPath = os.path.join(self.dataPath,self.image_list[index])

        Img = default_loader(dataPath)
        Img = self.transform(Img)

        return ImgA

    def __len__(self):
        return len(self.image_list)

