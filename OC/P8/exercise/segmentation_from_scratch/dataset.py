import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt  
from torchvision.transforms import ToPILImage  # To convert tensors to PIL images  

class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        # Get unique values and their counts  
        unique, counts = np.unique(mask, return_counts=True)  
        print("MASK", unique, counts)
        
        mask[mask == 255.0] = 1.0
        unique, counts = np.unique(mask, return_counts=True)  
        print("MASK 2", unique, counts)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        # to_pil = ToPILImage()  
        # pil_img = to_pil(image)  
        # pil_img.save("test.png")
        # pil_mask = to_pil(mask)  
        # pil_mask.save("mask.png")

        return image, mask
