import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt  
from torchvision.transforms import ToPILImage  # To convert tensors to PIL images  
import json

from helpers import *


class CityScapeDataset(Dataset):
    def __init__(self, dataframe,  transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        img_path = self.dataframe.loc[index]["original_img_path"]
        mask_path = self.dataframe.loc[index]["polygons_json_path"]

        img = Image.open(img_path).convert("RGB")
        image_np = np.array(img)

        mask =  Image.fromarray(polygon_to_mask_img(mask_path)).convert("RGBA")
        mask_np = np.array(mask.convert("L"), dtype=np.float32)

        # unique, counts = np.unique(mask_np, return_counts=True)  
        # print("MASK", unique, counts)
    
        if self.transform is not None:
            augmentations = self.transform(image=image_np, mask=mask_np)
            image_np = augmentations["image"]
            mask_np = augmentations["mask"]
        
        to_pil = ToPILImage()  
        pil_img = to_pil(image_np)  
        pil_img.save("img.png") 
        
        pil_img = to_pil(mask_np)  
        pil_img.save("mask.png") 

        return image_np, mask_np
