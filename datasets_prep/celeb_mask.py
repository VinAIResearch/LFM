import os

import albumentations
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


# With semantic map and scene label
class CelebAMask(Dataset):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic", crop_size=None):
        self.split = self.get_split()
        self.n_labels = 18  # unknown + 150
        self.image_root = "dataset/CelebAMask-HQ/CelebA-HQ-img"
        self.mask_root = "dataset/CelebAMask-HQ/mask"
        self._length = 27000 if self.split == "train" else 3000

        size = None if size is not None and size <= 0 else size
        self.size = size
        if crop_size is None:
            self.crop_size = size if size is not None else None
        else:
            self.crop_size = crop_size
        if self.size is not None:
            self.interpolation = interpolation
            self.interpolation = {
                "nearest": cv2.INTER_NEAREST,
                "bilinear": cv2.INTER_LINEAR,
                "bicubic": cv2.INTER_CUBIC,
                "area": cv2.INTER_AREA,
                "lanczos": cv2.INTER_LANCZOS4,
            }[self.interpolation]
            self.image_rescaler = albumentations.SmallestMaxSize(max_size=self.size, interpolation=self.interpolation)
            self.segmentation_rescaler = albumentations.SmallestMaxSize(
                max_size=self.size, interpolation=cv2.INTER_NEAREST
            )

        if crop_size is not None:
            self.center_crop = not random_crop
            if self.center_crop:
                self.cropper = albumentations.CenterCrop(height=self.crop_size, width=self.crop_size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.crop_size, width=self.crop_size)
            self.preprocessor = self.cropper

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        if self.split == "train":
            i = i % 27000
        else:
            i = i % 3000
            i = i + 27000
        image = Image.open(os.path.join(self.image_root, "{}.jpg".format(i)))
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        if self.size is not None:
            image = self.image_rescaler(image=image)["image"]

        segmentation = Image.open(os.path.join(self.mask_root, "{}.png".format(i)))
        segmentation = np.array(segmentation).astype(np.uint8)

        if segmentation.shape[-1] == 3:
            print(os.path.join(self.mask_root, "{}.png".format(i)))
            print(os.path.join(self.image_root, "{}.jpg".format(i)))

        if self.size is not None:
            segmentation = self.segmentation_rescaler(image=segmentation)["image"]
        if self.size is not None:
            processed = self.preprocessor(image=image, mask=segmentation)
            image = processed["image"]
            segmentation = processed["mask"]

        image = (image / 127.5 - 1.0).astype(np.float32)
        image = torch.from_numpy(image).permute(2, 0, 1)
        segmentation = torch.from_numpy(segmentation).long()
        # print(image.shape)
        # print(segmentation.shape)
        return image, segmentation


class CelebAMaskTrain(CelebAMask):
    # default to random_crop=True
    def __init__(self, size=None, random_crop=True, interpolation="bicubic", crop_size=None):
        super().__init__(size=size, random_crop=random_crop, interpolation=interpolation, crop_size=crop_size)

    def get_split(self):
        return "train"


class CelebAMaskValidation(CelebAMask):
    def get_split(self):
        return "validation"


if __name__ == "__main__":

    dataset = CelebAMaskTrain(size=256, crop_size=256)
    print(len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    image, segmentation = next(iter(dataloader))
    print(image.shape)
    print(torch.max(segmentation), torch.min(segmentation))
    pass
