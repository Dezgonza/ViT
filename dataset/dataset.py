import numpy as np
from PIL import Image
import torch, logging
from torch.utils.data import Dataset

class YogaDataset(Dataset):
    def __init__(self, imgs_dir, dtype='train', size=16*3):
        super().__init__()

        self.size = size
        self.imgs_dir = imgs_dir
        with open(imgs_dir + 'yoga_' + dtype + '_2.txt', 'r') as f:
            self.cls = f.readlines()
        f.close()
        logging.info(f'Creating dataset with {len(self.cls)} examples')

    def __len__(self):
        return len(self.cls)

    @classmethod
    def preprocess(cls, pil_img, size):
        assert size > 0, 'Scale is too small'
        pil_img = pil_img.resize((size, size))
        img_nd = np.array(pil_img)
        plt.imshow(img_nd)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    @classmethod
    def do_transpose(cls, img_nd):
        img_trans = img_nd.transpose((2, 0, 1))
        
        return img_trans

    def __getitem__(self, i):
        name, cls1, cls2, cls3 = self.cls[i].replace('\n', '').split(',')
        img_file = self.imgs_dir + name
        img = Image.open(img_file)
        img = self.preprocess(img, self.size)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'class': cls1
        }