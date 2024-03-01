from librairies import *

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

def display_loss_accuracy_dice_iou(history):    
    # Extracting values from history  
    loss = history.history['loss']    
    val_loss = history.history['val_loss']    
    acc = history.history['accuracy']    
    val_acc = history.history['val_accuracy']    
    dice_coef = history.history.get('dice_coef', [])  
    val_dice_coef = history.history.get('val_dice_coef', [])  
    iou = history.history.get('iou', [])  
    val_iou = history.history.get('val_iou', [])  
        
    _epochs = list(range(1, len(loss) + 1))    
        
    plt.figure(figsize=(12, 6))  
      
    # Plot for loss    
    plt.subplot(2, 2, 1)  # 2 rows, 2 columns, 1st subplot    
    plt.plot(_epochs, loss, 'y', label='Loss Train')    
    plt.plot(_epochs, val_loss, 'r', label='Loss Val')    
    plt.title('Loss Train & Val')    
    plt.xlabel('Epochs')    
    plt.ylabel('Loss')    
    plt.legend()    
        
    # Plot for accuracy    
    plt.subplot(2, 2, 2)  # 2 rows, 2 columns, 2nd subplot    
    plt.plot(_epochs, acc, 'y', label='Accuracy Train')    
    plt.plot(_epochs, val_acc, 'r', label='Accuracy Val')    
    plt.title('Accuracy Train & Val')    
    plt.xlabel('Epochs')    
    plt.ylabel('Accuracy')    
    plt.legend()  
      
    # Plot for Dice Coefficient  
    if dice_coef:  # Only plot if dice_coef exists  
        plt.subplot(2, 2, 3) # 2 rows, 2 columns, 3rd subplot  
        plt.plot(_epochs, dice_coef, 'g', label='Dice Coef Train')  
        plt.plot(_epochs, val_dice_coef, 'b', label='Dice Coef Val')  
        plt.title('Dice Coefficient Train & Val')  
        plt.xlabel('Epochs')  
        plt.ylabel('Dice Coefficient')  
        plt.legend()  
        
    # Plot for IoU  
    if iou:  # Only plot if iou exists  
        plt.subplot(2, 2, 4)  # 2 rows, 2 columns, 4th subplot  
        plt.plot(_epochs, iou, 'g', label='IoU Train')  
        plt.plot(_epochs, val_iou, 'b', label='IoU Val')  
        plt.title('IoU Train & Val')  
        plt.xlabel('Epochs')  
        plt.ylabel('IoU')  
        plt.legend()  
  
    plt.tight_layout()    
    plt.show()  
    
def show_grid_prediction(image, gt_mask, pred_mask, correct_pred_mask, cropped_image):
    fig, ax = plt.subplots(2, 3, figsize=(18, 6))  

    # Original image
    ax[0][0].imshow(image)  
    ax[0][0].set_title('Original Image')  
    ax[0][0].axis("off")
    
    # Ground thruth mask
    ax[0][1].imshow(gt_mask, cmap='viridis')  
    ax[0][1].set_title('Ground thruth mask')  
    ax[0][1].axis("off")
    
    # Predicted Mask
    ax[0][2].imshow(pred_mask)
    ax[0][2].set_title('Predicted Mask')  
    ax[0][2].axis("off")
    
    # Predicted Mask + Overlay
    ax[1][0].imshow(image)
    ax[1][0].imshow(pred_mask, cmap='viridis', alpha=0.5)  # 'alpha' controls the opacity  
    ax[1][0].set_title('Original Image + Pred Mask')  
    ax[1][0].axis("off")
    
    # Correct predicted Mask
    ax[1][1].imshow(correct_pred_mask)
    ax[1][1].set_title('Original Image + Pred Mask')  
    ax[1][1].axis("off")
    
    # Original Image Cropped
    ax[1][2].imshow(cropped_image)
    ax[1][2].set_title('Original Image + Pred Mask')  
    ax[1][2].axis("off")
    
    plt.tight_layout()
    plt.show()
    
def show_prediction(image, pred_mask):
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))  
      
    # Original image
    ax[0].imshow(image)  
    ax[0].set_title('Original Image')  
    ax[0].axis("off")
    
    # Predicted Mask
    ax[1].imshow(pred_mask)
    ax[1].set_title('Predicted Mask')  
    ax[1].axis("off")
    
    # Predicted Mask + Overlay
    ax[2].imshow(image)
    ax[2].imshow(pred_mask, cmap='viridis', alpha=0.5)  # 'alpha' controls the opacity  
    ax[2].set_title('Original Image + Pred Mask')  
    ax[2].axis("off")
    
    plt.tight_layout()
    plt.show()
    
def dice_coef(y_true, y_pred, smooth=1e-6):  
    y_true_f = K.flatten(y_true)  
    y_pred_f = K.flatten(y_pred)  
    intersection = K.sum(y_true_f * y_pred_f)  
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)  
  
def iou(y_true, y_pred, smooth=1e-6):  
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])  
    union = K.sum(y_true,[1,2,3]) + K.sum(y_pred,[1,2,3]) - intersection  
    return K.mean((intersection + smooth) / (union + smooth), axis=0)  
    
def display_images_from_tf_dataset(dataset, num_images=5):  
    # Make sure our dataset iterates only once through a portion of the data by using take  
    for images, masks in dataset.take(1):  # Taking 1 batch from the dataset  
        plt.figure(figsize=(10, 2 * num_images))  # Adjust figure size as needed  
        for i in range(num_images):  
            plt.subplot(num_images, 2, 2*i + 1)  
            plt.imshow(images[i].numpy())  # Convert to numpy and display the image  
            plt.title("Image")  
            plt.axis("off")  
            
            plt.subplot(num_images, 2, 2*i + 2)  
            plt.imshow(masks[i].numpy(), cmap='gray')  # Display the mask  
            plt.title("Mask")  
            plt.axis("off")  
        plt.tight_layout()  
        plt.show()  
        break 
    
    
def show_data_generator_images_sample(n):
    # Parameters  
    rows = 5  # 40 / 4  
    images_per_row = 4  
    total_images = rows * images_per_row  
    
    sample_generator_iter = iter(n.sample_generator)  
    
    # Adjust for 4 images and masks per row (8 columns in total)  
    fig, axs = plt.subplots(rows, images_per_row * 2, figsize=(20, round(1.5 * rows)))    
    
    for i in range(total_images):    
        image, mask = next(sample_generator_iter)    
        axs[i // images_per_row, (i % images_per_row) * 2].imshow(image[0])    
        axs[i // images_per_row, (i % images_per_row) * 2].axis('off')    
        axs[i // images_per_row, (i % images_per_row) * 2 + 1].imshow(tf.argmax(mask[0], axis=-1), cmap='viridis')    
        axs[i // images_per_row, (i % images_per_row) * 2 + 1].axis('off')    
        
    plt.tight_layout()    
    plt.show()    
    
def evaluate_with_and_without_aug(n, n_not_aug=None):
    test_loss, test_dice_coef, test_iou, test_accuracy = n.model.evaluate(n.test_generator)
    if n_not_aug:
        test_loss_not_aug, test_dice_coef_not_aug, test_iou_not_aug, test_accuracy_not_aug = n_not_aug.model.evaluate(n_not_aug.test_generator)  

    metrics = ['Loss', 'Dice Coefficient', 'IoU', 'Accuracy']  
    values = [test_loss, test_dice_coef, test_iou, test_accuracy] 
    if n_not_aug: 
        values_not_aug = [test_loss_not_aug, test_dice_coef_not_aug, test_iou_not_aug, test_accuracy_not_aug]  
    
    x = np.arange(len(metrics))  
    
    plt.figure(figsize=(12, 6))  
    bars1 = plt.bar(x - 0.2, values, width=0.4, color='skyblue', label='With Augmentation', align='center')
    if n_not_aug:   
        bars2 = plt.bar(x + 0.2, values_not_aug, width=0.4, color='orange', label='Without Augmentation', align='center')  
    
    plt.xlabel('Metrics', fontsize=14)  
    plt.ylabel('Values', fontsize=14)  
    plt.xticks(x, metrics, fontsize=12)  
    plt.yticks(fontsize=12)  
    plt.legend(fontsize=12, loc='upper left', bbox_to_anchor=(1.0, 0.15), fancybox=True, shadow=True)  
    
    plt.title('Model Performance Comparison: With vs Without Augmentation', fontsize=16)  
    
    # Adding value labels on top of each bar  
    def add_labels(bars):  
        for bar in bars:  
            height = bar.get_height()  
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01, f'{height:.2f}',  # Small y adjustment  
                    ha='center', va='bottom', fontsize=10)  
    
    add_labels(bars1) 
    if n_not_aug: 
        add_labels(bars2)  
    
    plt.grid(True)  
    plt.tight_layout()  
    plt.show()  