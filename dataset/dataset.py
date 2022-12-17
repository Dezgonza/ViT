import numpy as np
from PIL import Image
import torch, logging
from torch.utils.data import Dataset

class YogaDataset(Dataset):
    def __init__(self, imgs_dir, transform=None, ltype=3, dtype='train', size=16*3):
        super().__init__()

        self.size = size
        self.ltype = ltype
        self.imgs_dir = imgs_dir
        self.transform = transform
        with open(imgs_dir + 'yoga_' + dtype + '_2.txt', 'r') as f:
            self.cls = f.readlines()
        f.close()
        logging.info(f'Creating dataset with {len(self.cls)} examples')

    def __len__(self):
        return len(self.cls)

    def __getitem__(self, i):
        data = self.cls[i].replace('\n', '').split(',')
        name, cls = data[0], int(data[self.ltype])
        img = Image.open(self.imgs_dir + name)
        img = img.convert('RGBA' if img.info.get("transparency", False) else 'RGB')
        img = Image.fromarray(np.array(img)[:,:,:3])
        t_img = self.transform(img)
        img.close()

        return t_img, cls
