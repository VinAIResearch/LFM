import os
import cv2
import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class InpaintingTrainDataset(Dataset):
    def __init__(self, indir, mask_generator, transform):
        self.in_files = list(glob.glob(os.path.join(indir, '**', '*.jpg'), recursive=True))
        self.mask_generator = mask_generator
        self.transform = transform
        self.iter_i = 0

    def __len__(self):
        return len(self.in_files)

    def __getitem__(self, item):
        img = np.array(Image.open(self.in_files[item]).convert("RGB"))
        img = np.transpose(img, (2, 0, 1))
        mask = self.mask_generator(img, iter_i=self.iter_i)
        mask = torch.from_numpy(mask)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img)
        img = self.transform(img)
        masked_image = (1 - mask) * img
        self.iter_i += 1
        return dict(img=img, mask=mask, masked_image=masked_image)
        
        
def test():
    import torchvision.transforms as transform
    from inpaint_preprocess.mask import get_mask_generator
    import torchvision
    transforms = transform.Compose([
        transform.Normalize(0.5, 0.5)
    ])
    mask_gen = get_mask_generator(None, None)
    dataset = InpaintingTrainDataset(indir="dataset/data256x256/",
                                     mask_generator=mask_gen,
                                     transform=transforms)
    
    batch = dataset[0]
    
    mask = batch["mask"].reshape(256, 256)
    image = batch["img"]
    inpainting = batch["masked_image"]
    
    torchvision.utils.save_image(mask, "imask.png")
    torchvision.utils.save_image(image, "image.png")
    torchvision.utils.save_image(inpainting, "image_mask.png")
    
test()