import numpy as np
from PIL import Image
import torch, logging
from torch.utils.data import Dataset

class YogaDataset(Dataset):
    def __init__(self, imgs_dir, scale=1):

        self.imgs_dir = imgs_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.cls = None
        logging.info(f'Creating dataset with {len(self.cls)} examples')

    def __len__(self):
        return len(self.cls)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

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
        name = self.ids[i]
        img_file = list(self.imgs_dir.glob(name + '.*')) 

        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {name}: {img_file}'
 
        img = Image.open(img_file[0])
        img = self.preprocess(img, self.scale)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'class': 'holi'
        }