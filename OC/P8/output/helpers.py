import os

from glob import glob
import pandas as pd
import seaborn as sns  
import matplotlib.pyplot as plt  
from PIL import Image
from pandarallel import pandarallel  
import torchvision.transforms.functional as TF  
import torch
import json
import numpy as np
import cv2
import json


# Category reduction mapping
category_mapping = {  
    # flat  
    "road": 1, "sidewalk": 1, "parking": 1, "rail track": 1,  
    # human  
    "person": 2, "rider": 2, "persongroup": 2,  
    # vehicle  
    "car": 3, "truck": 3, "bus": 3, "on rails": 3, "motorcycle": 3, "bicycle": 3,  "bicyclegroup": 3, "caravan": 3, "trailer": 3,
    "ego vehicle": 3, "cargroup": 3, "license plate": 3, "train": 3, "motorcyclegroup": 3, "ridergroup": 3, "truckgroup": 3, "rectification border": 3,
    # construction  
    "building": 4, "wall": 4, "fence": 4, "guard rail": 4, "bridge": 4, "tunnel": 4,  
    # object  
    "pole": 5, "polegroup": 5, "traffic sign": 5, "traffic light": 5,  
    # nature  
    "vegetation": 6, "terrain": 6,  
    # sky  
    "sky": 7,  
    # void  
    "ground": 8, "dynamic": 8, "static": 8, "out of roi": 8
}

def polygon_to_mask_img(json_path):
    
    # Load the JSON File
    json_data = json.load(open(json_path))
    
    # Create the mask image array
    mask = np.zeros((json_data["imgHeight"], json_data["imgWidth"]), dtype=np.uint8)  
    
    # Set the polygons within the mask
    for obj in json_data['objects']:  
        label = obj['label']  
        polygon = np.array(obj['polygon'], np.int32)
        
        category_nb = category_mapping[label]
        if category_nb is not None:    
            cv2.fillPoly(mask, [polygon], category_nb)  # Fill with category number instead of color  
        else:    
            print(f"Label {label} not found in category mapping.")
        
    return mask

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 256



def check_accuracy(loader, model, device="mps"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0 # Better evaluation
    model.eval()
    
    with torch.no_grad():
        
        for x,y in loader:
            x = x.to(device)
            
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )
    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(
        f"dice_score {dice_score/len(loader)}"
    )
    
    model.train()