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
    



def get_file_info(json_path):
    file_path_prefix = json_path.replace("_polygons.json", "")
    
    original_img_path = (
        file_path_prefix
            .replace("gtFine/","leftImg8bit/")
            .replace("_gtFine", "_leftImg8bit.png")
    )
    
    color_img_path = file_path_prefix + "_color.png"
    instance_ids_img_path = file_path_prefix + "_instanceIds.png"
    label_ids_img_path = file_path_prefix + "_labelIds.png"
    polygons_json_path = json_path
    
    return [original_img_path, color_img_path, instance_ids_img_path, label_ids_img_path, polygons_json_path]

def get_dataframes(data_path, sample_dataset=0):
    json_paths_train = glob(f'{data_path}/train/**/*.json')
    json_paths_test = glob(f'{data_path}/test/**/*.json')
    json_paths_val = glob(f'{data_path}/val/**/*.json')

    data_train = [get_file_info(path) for path in json_paths_train]
    data_test = [get_file_info(path) for path in json_paths_test]
    data_val = [get_file_info(path) for path in json_paths_val]

    columns = ["original_img_path", "color_img_path", "instance_ids_img_path", "label_ids_img_path", "polygons_json_path"]
    df_train = pd.DataFrame(data_train, columns=columns)
    df_test = pd.DataFrame(data_test, columns=columns)
    df_val = pd.DataFrame(data_val, columns=columns)
    
    if sample_dataset > 0:
        print(f"Sampling the dataset to {sample_dataset}.")
        df_train = df_train.sample(n=sample_dataset, random_state=42).reset_index(drop=True)
        # df_test = df_test.sample(n=sample_dataset, random_state=42).reset_index(drop=True)
        df_val = df_val.sample(n=100, random_state=42).reset_index(drop=True)
    else:
        print("Using the full dataset")
    
    return df_train, df_test, df_val