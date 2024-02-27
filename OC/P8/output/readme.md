Metrics:
Intersection-over-Union (IoU)
Average Precision
MEan Average Precision


SAM > https://github.com/huggingface/notebooks/blob/main/examples/segment_anything.ipynb
Transformers tutorials Segformer > https://github.com/NielsRogge/Transformers-Tutorials/tree/master/SegFormer

```py

class DataGeneratorVisualization(DataGenerator):  
    def visualize_augmentations(self, index=0):  
        """Visualize augmentations for the dataset item at the given index."""  
        # Load the original image and mask  
        image_path = self.image_paths[index]  
        mask_path = self.mask_paths[index]  
          
        img = cv2.imread(image_path)  
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  
          
        if self.augmentations:  
            # Apply the augmentations  
            augmented = self.augmentations(image=img, mask=mask)  
            img_aug = augmented['image']  
            mask_aug = augmented['mask']  
        else:  
            img_aug = img.copy()  
            mask_aug = mask.copy()  
  
        fig, axs = plt.subplots(2, 2, figsize=(10, 7))  
          
        axs[0, 0].imshow(img)  
        axs[0, 0].set_title('Original Image')  
        axs[1, 0].imshow(mask, cmap='gray')  
        axs[1, 0].set_title('Original Mask')  
          
        axs[0, 1].imshow(img_aug)  
        axs[0, 1].set_title('Augmented Image')  
        axs[1, 1].imshow(mask_aug, cmap='gray')  
        axs[1, 1].set_title('Augmented Mask')  
          
        for ax in axs.flat:  
            ax.axis('off')  
          
        plt.tight_layout()  
        plt.show()  
To use this method, initialize an instance of DataGeneratorVisualization just like you would with DataGenerator, then call visualize_augmentations(index) for any index within your data:

# Assuming the `train_img_paths` and `train_mask_paths` are already defined  
# and `augmentations` is your albumentations.Compose object from earlier  
  
train_generator_viz = DataGeneratorVisualization(  
    image_paths=train_img_paths,  
    mask_paths=train_mask_paths,  
    batch_size=32,  # The batch size doesn't matter for visualization  
    dim=(256, 256), # Adjust dimensions as needed  
    n_classes=8,    # Adjust number of classes as needed  
    shuffle=True,  
    augmentations=augmentations  
)  
  
train_generator_viz.visualize_augmentations(index=0)  # Vis
```