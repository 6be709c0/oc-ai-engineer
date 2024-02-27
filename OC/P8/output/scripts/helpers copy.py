from librairies import *
from config import *
from download_dataset import *
from notebook_class import *

def print_config():
    display(HTML('<h1 style="padding:0;margin:0;"><b>CONFIG USED</b></h1>'))  
    print(json.dumps(config, indent=4),"\n")

def prepare_notebook(config):
    print_config(config)
    download_dataset(config)
    unzip_dataset(config)
    return get_dataframes(config)


categories = {  
    'void': [0, 1, 2, 3, 4, 5, 6],  
    'flat': [7, 8, 9, 10],  
    'construction': [11, 12, 13, 14, 15, 16],  
    'object': [17, 18, 19, 20],  
    'nature': [21, 22],  
    'sky': [23],  
    'human': [24, 25],  
    'vehicle': [26, 27, 28, 29, 30, 31, 32, 33, -1]  
}  
# Corrected Category Mapping  
category_mapping = {    
    # void    
    "ground": 0, "dynamic": 0, "static": 0, "out of roi": 0, "ego vehicle": 0,
    # flat    
    "road": 1, "sidewalk": 1, "parking": 1, "rail track": 1,  
     # construction    
    "building": 2, "wall": 2, "fence": 2, "guard rail": 2, "bridge": 2, "tunnel": 2,    
      # object    
    "pole": 3, "polegroup": 3, "traffic sign": 3, "traffic light": 3,    
     # nature    
    "vegetation": 4, "terrain": 4,    
    # sky    
    "sky": 5,    
    # human    
    "person": 6, "rider": 6, "persongroup": 6,    
    # vehicle    
    "car": 7, "truck": 7, "bus": 7, "on rails": 7, "motorcycle": 7, "bicycle": 7,  "bicyclegroup": 7, "caravan": 7, "trailer": 7,  
    "cargroup": 7, "license plate": 7, "train": 7, "motorcyclegroup": 7, "ridergroup": 7, "truckgroup": 7, "rectification border": 7,  
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

def get_dataframes(config):
    
    sample_dataset = config["train_sample_nb"]
    
    json_paths_train = glob(f'{config["data_path"]}/train/**/*.json')
    json_paths_test = glob(f'{config["data_path"]}/test/**/*.json')
    json_paths_val = glob(f'{config["data_path"]}/val/**/*.json')

    data_train = [get_file_info(path) for path in json_paths_train]
    data_test = [get_file_info(path) for path in json_paths_test]
    data_val = [get_file_info(path) for path in json_paths_val]

    columns = ["original_img_path", "color_img_path", "instance_ids_img_path", "label_ids_img_path", "polygons_json_path"]
    df_train = pd.DataFrame(data_train, columns=columns)
    df_test = pd.DataFrame(data_test, columns=columns)
    df_val = pd.DataFrame(data_val, columns=columns)
    
    if sample_dataset > 0:
        print(f"\nSampling the training dataset from {df_train.shape[0]} to {sample_dataset}.")
        df_train = df_train.sample(n=sample_dataset, random_state=42).reset_index(drop=True)
        # df_test = df_test.sample(n=sample_dataset, random_state=42).reset_index(drop=True)
        # df_val = df_val.sample(n=100, random_state=42).reset_index(drop=True)
    else:
        print("Using the full dataset")
    
    return df_train, df_test, df_val


def read_overlay(image_path, mask_path, mask_opacity=0.5):  
    """  
    Display the mask on top of the image with an opacity of 0.5.  
      
    Parameters:  
    image_path (str): Path to the image file.  
    mask_path (str): Path to the mask file (in your case, path which will be processed by polygon_to_mask_img).  
    mask_opacity (float, optional): Opacity of the mask. Default is 0.5.  
      
    Returns:  
    numpy.ndarray: The overlaid image.  
    """  
      
    # Read the image and the mask  
    image = read_image(image_path)  
    mask = read_mask(mask_path)  
      
    # Convert the image to a format that can be handled by addWeighted (needs scaling up to 255 and converting to uint8)  
    image_for_overlay = (image * 255).astype(np.uint8)  
      
    # Blend the image and the mask  
    # mask_opacity for the mask weight, and 1-mask_opacity for the image to ensure they sum to 1 for proper blending  
    overlay_result = cv2.addWeighted(image_for_overlay, 1-mask_opacity, mask, mask_opacity, 0)  
      
    return overlay_result  


# Configurable dimensions: Update these based on your requirement or config 
def read_image(x, config):  
    x = cv2.imread(x, cv2.IMREAD_COLOR)  
    x = cv2.resize(x, (config["width"], config["height"]))  
    x = x/255.0  
    x = x.astype(np.float32)  
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    return x  
  
def read_mask(path, config):  
    mask_img_array = polygon_to_mask_img(path)
    color_map = mpl.colormaps.get_cmap('viridis')  
      
    mask_img = cv2.resize(mask_img_array, (config["width"], config["height"]))  
    return mask_img.astype(np.int32)

def read_mask_for_overlay(path, config):  
    mask_img_array = polygon_to_mask_img(path)  
    color_map = mpl.colormaps.get_cmap('viridis')  
              
    mask_img_array_normalized = mask_img_array / mask_img_array.max()  
    mask_img_colored = color_map(mask_img_array_normalized)[:, :, :3]  # Exclude the alpha channel  
    mask_img_colored = (mask_img_colored * 255).astype(np.uint8)  
      
    mask_img = cv2.resize(mask_img_colored, (config["width"], config["height"]))  
    return mask_img  

def read_overlay(row, mask_opacity=0.5):  
    image = read_image(row["original_img_path"])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = read_mask_for_overlay(row["polygons_json_path"])  
      
    overlayed = cv2.addWeighted(src1=image, alpha=(1-mask_opacity), src2=mask.astype(np.float32)/255.0, beta=mask_opacity, gamma=0)  
    
    return overlayed

def tf_dataset(x,y, batch=4):
    dataset = tf.data.Dataset.from_tensor_slices((x,y)) # Dataset data object created from input and target data
    dataset = dataset.shuffle(buffer_size=100) ## selected from the first 100 samples
    dataset = dataset.map(preprocess) # Applying preprocessing to every batch in the Dataset object
    dataset = dataset.batch(batch) # Determine batch-size
    dataset = dataset.repeat()
    dataset = dataset.prefetch(2) # Optimization to reduce waiting time on each object
    return dataset
        

def preprocess(x,y):
    def f(x,y):
        x = x.decode() ##byte stream conversion
        y = y.decode()
        image = read_image(x)
        mask = read_mask(y)
        return image, mask
    
    image, mask = tf.numpy_function(f,[x,y],[tf.float32, tf.int32])
    mask = tf.one_hot(mask, config["classes"], dtype=tf.int32)
    
    image.set_shape([config["height"], config["width"], 3])    
    mask.set_shape([config["height"], config["width"], config["classes"]])
    
    return image, mask
