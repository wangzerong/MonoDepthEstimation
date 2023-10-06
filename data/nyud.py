import cv2

from PIL import Image
import numpy as np
import torch.utils.data as data
import os
import json
class NYUD_TRAIN_DATASET(data.Dataset):
    def __init__(self, args,  filepath = None, transform = None):
        if not os.path.exists(filepath):
            raise ValueError(f"Check the filepath value:{filepath}")
        self.filepath = filepath
        self.lines = None
        self.scale = args.scale
        self.transform = transform

        with open(self.filepath, "r") as f:
            self.lines = f.readlines()
    
    def __getitem__(self,index):
        line = self.lines[index].strip()
        info = json.loads(line)
        image_info, depth_info = info["image_info"], info["depth_info"]
        image_path, image_format = image_info["image_path"], image_info["format"]
        depth_path, depth_format = depth_info["depth_path"], depth_info["format"]
        focal = info["focal"]

        if image_format == "jpg":
            img = Image.open(image_path).convert("RGB")
            img = np.array(img, dtype=np.float32, copy=False)
        
        if depth_format == "png":
            depth = cv2.imread(depth_path, -1 )/1000.0
        sample = {"image":img ,
                  "depth":depth,
                  "meta":self.scale
        }
        if not self.transform:
            sample = self.transform(sample)
        img, depth, scale = sample["image"], sample["depth"], sample["meta"]
        return img

    def __len__(self):
        return len(self.lines)