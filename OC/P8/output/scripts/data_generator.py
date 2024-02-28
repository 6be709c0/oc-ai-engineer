import numpy as np  
import cv2  
from tensorflow.keras.utils import Sequence  
import tensorflow as tf
class DataGenerator(Sequence):  
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
    
    def __init__(self, img_paths, mask_paths, batch_size, config, shuffle=True, augmentation=None):  
        self.img_paths = img_paths  
        self.mask_paths = mask_paths  
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.cfg = config  
        self.shuffle = shuffle  
        self.indexes = np.arange(len(self.img_paths))  
        if self.shuffle:  
            np.random.shuffle(self.indexes)  
            
        self.category_ids = {cat: idx for idx, cat in enumerate(self.categories.keys())} 
        
  
    def __len__(self):  
        # Compute the number of batches to cover all samples  
        return int(np.floor(len(self.img_paths) / self.batch_size))  
  
    def __getitem__(self, index):  
        # Generate indexes for the batch  
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]  
          
        # Generate data  
        X, y = self.__data_generation(batch_indexes)  
          
        return X, y  
  
    def on_epoch_end(self):  
        # Shuffle indexes at the end of each epoch  
        if self.shuffle:  
            np.random.shuffle(self.indexes)  
  
    def __data_generation(self, batch_indexes):  
        # Initialization  
        if self.augmentation:  
            X = np.empty((self.batch_size * self.cfg["augment_per_image"],self.cfg["height"],self.cfg["width"], 3))  
            y = np.empty((self.batch_size * self.cfg["augment_per_image"],self.cfg["height"],self.cfg["width"], self.cfg["classes"]), dtype=int)  
        else:
            X = np.empty((self.batch_size, self.cfg["height"],self.cfg["width"], 3))  
            y = np.empty((self.batch_size, self.cfg["height"],self.cfg["width"], self.cfg["classes"]), dtype=int)  
        
        ctr = 0
        # Generate data  
        for i, idx in enumerate(batch_indexes):  
            img_path = self.img_paths[idx]  
            mask_path = self.mask_paths[idx]  
  
            image = self.__read_image(img_path)  
            mask = self.__read_mask(mask_path)  
            
            if self.augmentation:
                for n in range(self.cfg["augment_per_image"]):
                    # Apply augmentations  
                    augmented = self.augmentation(image=image, mask=mask)  
                    aug_image = augmented['image']  
                    aug_mask = augmented['mask']
                    aug_mask = tf.one_hot(aug_mask, self.cfg["classes"], dtype=tf.int32).numpy()
                    X[ctr,] = aug_image
                    y[ctr,] = aug_mask
                    ctr +=1
                
                # # Add the original image as well
                # mask = tf.one_hot(mask, self.cfg["classes"], dtype=tf.int32).numpy()
                # X[ctr,] = image
                # y[ctr,] = mask
                
            else:
                mask = tf.one_hot(mask, self.cfg["classes"], dtype=tf.int32).numpy()
                X[i,] = image
                y[i,] = mask
                
        return X, y
  
    # You can reuse the read_image and read_mask methods from NotebookProcessor here  
    # or implement them directly inside this class.  
    def __read_image(self, img_path):  
        x = cv2.imread(img_path, cv2.IMREAD_COLOR)  
        x = cv2.resize(x, (self.cfg["width"], self.cfg["height"]))  
        x = x/255.0  
        x = x.astype(np.float32)  
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        
        return x  
  
    def __read_mask(self, mask_path):  
        original_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)   
        mask = np.zeros_like(original_mask, dtype=np.int32)  

        # Map each label in the original mask to its new category  
        for original_label, new_category in self.categories.items():  
            for label in new_category:  
                if label == -1:  # Handle the case where label is -1  
                    mask[original_mask == 255] = self.category_ids[original_label]  # Assuming 255 is used for label -1 in your dataset  
                else:  
                    mask[original_mask == label] = self.category_ids[original_label]  

        # Resize the mask after the mapping
        mask = cv2.resize(mask, (self.cfg["width"], self.cfg["height"]), interpolation=cv2.INTER_NEAREST)  

        return mask  