from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import requests
import torch
import os

def load_and_resize_image(path, target_size):
    return Image.open(path).resize(target_size)

class CustomDataset(Dataset):
    def __init__(self, processor, img_dir, mask_dir, label2id, target_size):
        self.processor = processor
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.target_size = target_size
        self.label2id   = label2id               # store it
        self.length = len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        #fn = self.filenames[idx]
        img_arrays = sorted(os.listdir(self.img_dir))
        mask_arrays = sorted(os.listdir(self.mask_dir))
        filImg = img_arrays[idx]
        filMask = mask_arrays[idx]


        img_url = os.path.join(self.img_dir, filImg)
        image = load_and_resize_image(img_url, self.target_size)
        image_array = np.array(image)

        # load semantic segmentation map similarly
        map_url = os.path.join(self.mask_dir, filMask)
        map_image = load_and_resize_image(map_url, self.target_size)
        map_array = np.array(map_image.convert("L"), dtype=np.int64)

        remapped = np.zeros_like(map_array)
        for orig_id, new_id in self.label2id.items():
            remapped[map_array == orig_id] = new_id

        # use processor to convert this to the required inputs
        inputs = self.processor(
            images=image_array,
            segmentation_maps=remapped,
            task_inputs=["semantic"],
            return_tensors="pt"
        )
        inputs = {k: v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k, v in inputs.items()}
        # Move inputs to GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
   
        return inputs

    def __len__(self):
        return self.length