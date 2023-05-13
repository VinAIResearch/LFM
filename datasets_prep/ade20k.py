import os
import numpy as np
import cv2
import albumentations
from PIL import Image
from torch.utils.data import Dataset
import torch
from datasets_prep.coco import SegmentationBase # for examples included in repo
import torchvision.transforms as transforms

class Examples(SegmentationBase):
    def __init__(self, size=256, random_crop=False, interpolation="bicubic"):
        super().__init__(data_csv="data/ade20k_examples.txt",
                         data_root="data/ade20k_images",
                         segmentation_root="data/ade20k_segmentations",
                         size=size, random_crop=random_crop,
                         interpolation=interpolation,
                         n_labels=151, shift_segmentation=False)


# With semantic map and scene label
class ADE20kBase(Dataset):
    def __init__(self, config=None, size=None, random_crop=False, interpolation="bicubic", crop_size=None):
        self.split = self.get_split()
        self.n_labels = 151 # unknown + 150
        self.data_csv = {"train": "dataset/ade20k/training.txt",
                         "validation": "dataset/ade20k/validation.txt"}[self.split]
        self.data_root = "dataset/ade20k"
        with open(os.path.join(self.data_root, "sceneCategories.txt"), "r") as f:
            self.scene_categories = f.read().splitlines()
        self.scene_categories = dict(line.split() for line in self.scene_categories)
        with open(self.data_csv, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.labels = {
            "file_path_": [os.path.join(self.data_root, l.split(" ")[0]) for l in self.image_paths],
            "segmentation_path_": [os.path.join(self.data_root, l.split(" ")[1]) for l in self.image_paths]
        }
        
        self.size = size
        self.image_transforms = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])
        
        self.segmentation_transforms = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
        ])

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        print(i)
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = self.image_transforms(image)
        segmentation = Image.open(example["segmentation_path_"])
        segmentation = self.segmentation_transforms(segmentation)
        segmentation = np.asarray(segmentation)

        example["image"] = image
        onehot = np.eye(self.n_labels)[segmentation]
        example["segmentation"] = torch.from_numpy(onehot)
        
        return example


class ADE20kTrain(ADE20kBase):
    # default to random_crop=True
    def __init__(self, config=None, size=None, random_crop=True, interpolation="bicubic", crop_size=None):
        super().__init__(config=config, size=size, random_crop=random_crop,
                          interpolation=interpolation, crop_size=crop_size)

    def get_split(self):
        return "train"


class ADE20kValidation(ADE20kBase):
    def get_split(self):
        return "validation"


if __name__ == "__main__":
    dset = ADE20kValidation()
    ex = dset[0]
    for k in ["image", "scene_category", "segmentation"]:
        print(type(ex[k]))
        try:
            print(ex[k].shape)
        except:
            print(ex[k])