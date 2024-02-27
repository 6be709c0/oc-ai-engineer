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

def polygon_to_mask_img(json_path, category_mapping):
    
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

def display_loss_and_accuracy(history):  
    loss = history.history['loss']  
    val_loss = history.history['val_loss']  
    acc = history.history['accuracy']  
    val_acc = history.history['val_accuracy']  
      
    _epochs = list(range(1, len(loss) + 1))  
      
    plt.figure(figsize=(12, 3))
      
    # Plot for loss  
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot  
    plt.plot(_epochs, loss, 'y', label='Loss train')  
    plt.plot(_epochs, val_loss, 'r', label='Loss val')  
    plt.title('Loss Train & Val')  
    plt.xlabel('Epochs')  
    plt.ylabel('Loss')  
    plt.legend()  
      
    # Plot for accuracy  
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot  
    plt.plot(_epochs, acc, 'y', label='Accuracy train')  
    plt.plot(_epochs, val_acc, 'r', label='Accuracy val')  
    plt.title('Accuracy Train & Val')  
    plt.xlabel('Epochs')  
    plt.ylabel('Accuracy')  
    plt.legend()  
      
    plt.tight_layout()  
    plt.show()  
    
    
def show_grid_prediction(image, gt_mask, pred_mask, correct_pred_mask, cropped_image, accuracy):
    fig, ax = plt.subplots(2, 3, figsize=(18, 6))  
    plt.suptitle(f'Accuracy {accuracy*100:.2f}%', fontsize=16, fontweight='bold', y=1.01)
      
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

def display_loss_accuracy_dice_iou(history):  
    loss = history.history['loss']  
    val_loss = history.history['val_loss']  
    dice = history.history['dice_coef']  
    val_dice = history.history['val_dice_coef']  
    iou = history.history['iou']  
    val_iou = history.history['val_iou']  
      
    _epochs = range(1, len(loss) + 1)  
      
    plt.figure(figsize=(16, 4))  # Increase the figure size for better visibility  
      
    # Plot for loss  
    plt.subplot(1, 3, 1)  # 1 row, 4 columns, 1st subplot  
    plt.plot(_epochs, loss, 'y', label='Training Loss')  
    plt.plot(_epochs, val_loss, 'r', label='Validation Loss')  
    plt.title('Training & Validation Loss')  
    plt.xlabel('Epochs')  
    plt.ylabel('Loss')  
    plt.legend()  
      
    # Plot for Dice Coefficient  
    plt.subplot(1, 3, 2)  # 1 row, 4 columns, 3rd subplot  
    plt.plot(_epochs, dice, 'y', label='Training Dice Coef')  
    plt.plot(_epochs, val_dice, 'r', label='Validation Dice Coef')  
    plt.title('Training & Validation Dice Coefficient')  
    plt.xlabel('Epochs')  
    plt.ylabel('Dice Coeff')  
    plt.legend()  
      
    # Plot for IoU  
    plt.subplot(1, 3, 3)  # 1 row, 4 columns, 4th subplot  
    plt.plot(_epochs, iou, 'y', label='Training IoU')  
    plt.plot(_epochs, val_iou, 'r', label='Validation IoU')  
    plt.title('Training & Validation IoU')  
    plt.xlabel('Epochs')  
    plt.ylabel('IoU')  
    plt.legend()  
      
    plt.tight_layout()  # Automatically adjust subplot params to give specified padding  
    plt.show()  
    
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