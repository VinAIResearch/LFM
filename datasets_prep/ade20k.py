import os

import albumentations
import cv2
import numpy as np
import torch
from datasets_prep.coco import SegmentationBase  # for examples included in repo
from PIL import Image
from torch.utils.data import Dataset


class Examples(SegmentationBase):
    def __init__(self, size=256, random_crop=False, interpolation="bicubic"):
        super().__init__(
            data_csv="data/ade20k_examples.txt",
            data_root="data/ade20k_images",
            segmentation_root="data/ade20k_segmentations",
            size=size,
            random_crop=random_crop,
            interpolation=interpolation,
            n_labels=151,
            shift_segmentation=False,
        )


# With semantic map and scene label
class ADE20kBase(Dataset):
    def __init__(self, config=None, size=None, random_crop=False, interpolation="bicubic", crop_size=None):
        self.split = self.get_split()
        self.n_labels = 151  # unknown + 150
        self.data_csv = {"train": "dataset/ade20k/training.txt", "validation": "dataset/ade20k/validation.txt"}[
            self.split
        ]
        self.data_root = "dataset/ade20k"
        with open(os.path.join(self.data_root, "sceneCategories.txt"), "r") as f:
            self.scene_categories = f.read().splitlines()
        self.scene_categories = dict(line.split() for line in self.scene_categories)
        with open(self.data_csv, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.labels = {
            "file_path_": [os.path.join(self.data_root, l.split(" ")[0]) for l in self.image_paths],
            "segmentation_path_": [os.path.join(self.data_root, l.split(" ")[1]) for l in self.image_paths],
        }

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
        # print(i)
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        if self.size is not None:
            image = self.image_rescaler(image=image)["image"]
        segmentation = Image.open(example["segmentation_path_"])
        segmentation = np.array(segmentation).astype(np.uint8)
        if self.size is not None:
            segmentation = self.segmentation_rescaler(image=segmentation)["image"]
        if self.size is not None:
            processed = self.preprocessor(image=image, mask=segmentation)
            image = processed["image"]
            segmentation = processed["mask"]

        image = (image / 127.5 - 1.0).astype(np.float32)
        image = torch.from_numpy(image).permute(2, 0, 1)
        segmentation = torch.from_numpy(segmentation).long()
        return image, segmentation


class ADE20kTrain(ADE20kBase):
    # default to random_crop=True
    def __init__(self, config=None, size=None, random_crop=True, interpolation="bicubic", crop_size=None):
        super().__init__(
            config=config, size=size, random_crop=random_crop, interpolation=interpolation, crop_size=crop_size
        )

    def get_split(self):
        return "train"


class ADE20kValidation(ADE20kBase):
    def get_split(self):
        return "validation"


if __name__ == "__main__":
    pass
    # dataset = ADE20kTrain(config=None, size=256, crop_size=256)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    # image, segmentation = next(iter(dataloader))
    # print(image.shape)
    # print(segmentation.shape)
