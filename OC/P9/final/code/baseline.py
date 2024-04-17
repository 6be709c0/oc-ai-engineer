import os  
import math  
import shutil  
import numpy as np

import tensorflow as tf  
from tensorflow.keras.preprocessing.image import ImageDataGenerator  
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam  
from sklearn.metrics import f1_score  

import matplotlib.pyplot as plt  
from sklearn.metrics import confusion_matrix  

# Set seeds for numpy and tensorflow   
np.random.seed(42)  
tf.random.set_seed(42) 

class Baseline:
    
    classes = ["anomaly","good"]
    batch_size = 32
    
    def __init__(self, config):
        
        self.cfg = config
        
        self.cfg["augmented_dataset_path"] = f"{self.cfg['dataset_path']}/augmented"
        self.cfg["augmented_test_dataset_path"] = f"{self.cfg['dataset_path']}/augmented_test"
        
        self.cfg["train_dataset_path"] = f"{self.cfg['dataset_path']}/train"
        self.cfg["test_dataset_path"] = f"{self.cfg['dataset_path']}/test"
        
    def augment_dataset(self, source_path, dest_path, multiplicator):
        print(f"\n- Preparing dataset {source_path} to {dest_path}")
        print("- Cleaning existing augmented directory")
        # Clean up the augmented directory
        if os.path.exists(dest_path):  
            shutil.rmtree(dest_path)
        
        print("- Creating augmented directory and subdirectories good and anomaly")
        # Create the augmented directory
        os.makedirs(dest_path)
        os.makedirs(f"{dest_path}/good")
        os.makedirs(f"{dest_path}/anomaly")
        
        # Initialize the image data generator for augmentation  
        self.datagen = ImageDataGenerator(  
            rescale=1./255,  
            rotation_range=40,  
            width_shift_range=0.2,  
            height_shift_range=0.2,  
            shear_range=0.2,  
            zoom_range=0.2,  
            horizontal_flip=True,  
            fill_mode='nearest'  
        )
        
        print(f"- Augmenting images (x{multiplicator})... ")
        # Apply augmentations and save images
        for cls in self.classes:  
            self.save_augmented_images(cls, source_path, dest_path, multiplicator)
            
        print("- Done")
        
    def prepare_dataset(self):
        
        self.augment_dataset(self.cfg["train_dataset_path"], self.cfg["augmented_dataset_path"], self.cfg['augmentation_multiplicator'])
        self.augment_dataset(self.cfg["test_dataset_path"], self.cfg["augmented_test_dataset_path"], self.cfg['augmentation_test__multiplicator'])

        # Load the augmented dataset  
        train_datagen = ImageDataGenerator(  
            rescale=1./255,  
            validation_split=0.2  # using 20% of the data for validation  
        )  
        
        self.train_generator = train_datagen.flow_from_directory(  
            self.cfg["augmented_dataset_path"],  
            target_size=(self.cfg['img_width'], self.cfg['img_height']),  
            batch_size=self.batch_size,  
            class_mode='categorical',  
            subset='training'  
        )  
        
        self.validation_generator = train_datagen.flow_from_directory(  
            self.cfg["augmented_dataset_path"],  
            target_size=(self.cfg['img_width'], self.cfg['img_height']),  
            batch_size=self.batch_size,  
            class_mode='categorical',  
            subset='validation'  
        ) 

        test_datagen = ImageDataGenerator(rescale=1./255)  
        
        self.test_generator = test_datagen.flow_from_directory(  
            directory=self.cfg["augmented_test_dataset_path"],  # Make sure this points to your test dataset directory  
            target_size=(self.cfg['img_width'], self.cfg['img_height']),  
            batch_size=self.batch_size,
            class_mode='categorical'  # or 'categorical' if you have more than two classes  
        )
        
        self.steps_per_epoch = np.ceil(self.train_generator.samples / self.batch_size)  
        self.validation_steps = np.ceil(self.validation_generator.samples / self.batch_size)  

        
    def save_augmented_images(self, class_name, source_path, dest_path, multiplicator):
        image_dir = os.path.join(source_path, class_name)
        images = [img for img in os.listdir(image_dir) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]  

        for img_name in images:  
            img_path = os.path.join(source_path, class_name, img_name)  
            img = load_img(img_path)  
            img = img.resize((self.cfg['img_width'], self.cfg['img_height']))  
            x = img_to_array(img)  
            x = np.expand_dims(x, axis=0)  
            i = 0  
            for _ in self.datagen.flow(x, batch_size=1,  
                                save_to_dir=os.path.join(dest_path, class_name),  
                                save_prefix='aug',  
                                save_format='jpeg'):  
                i += 1  
                if i >= multiplicator:
                    break
                
    def display_samples(self, generator, num_images=8):
        # Get a batch of images  
        imgs, labels = next(generator)  # 'next' retrieves the next batch  
        
        # Determine the grid size needed  
        nrows = int(num_images ** 0.5)  
        ncols = int(np.ceil(num_images / nrows))  
        
        # Set up matplotlib fig, and size it to fit `num_images` images  
        plt.figure(figsize=(8, 4))  
        for i in range(num_images):  
            plt.subplot(nrows, ncols, i + 1)  
            # Rescale images to display them correctly  
            img = imgs[i] * 255  # Assuming the generator rescales images by 1/255  
            plt.imshow(img.astype(np.uint8))  
            
            # Adjusting label display for possibly one-hot encoded labels  
            if generator.class_mode == 'categorical':  
                # For one-hot encoded labels, find the index of the max value  
                label = np.argmax(labels[i])  
                if label == 0:  
                    plt.title('Anomaly')  
                else:  
                    plt.title('Good')  
            else:  
                # For binary labels  
                if labels[i] == 1:  
                    plt.title('Good')  
                else:  
                    plt.title('Anomaly')  
    
            plt.axis('off')  
        plt.tight_layout()  
        plt.show()  
        
        # 
    def create_model(self):
        # Initializing MobileNet with include_top=False to customize the output layers  
        base_model = MobileNet(weights='imagenet', include_top=False,  
                            input_shape=(self.cfg['img_width'], self.cfg['img_height'], 3))

        for layer in base_model.layers[-20:]:
            if not isinstance(layer, layers.BatchNormalization):  
                layer.trainable = True

        self.model = models.Sequential([  
            base_model,  
            layers.GlobalAveragePooling2D(),  
            layers.Dense(256, activation='relu'),  
            layers.Dropout(0.5),  
            layers.Dense(2, activation='softmax') # Output layer for two classes  
        ])
        
        self.model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',  # Corrected loss function  
              metrics=['accuracy'])
        
    def fit_model(self):
        self.history = self.model.fit(  
            self.train_generator,  
            steps_per_epoch=self.steps_per_epoch,  
            validation_data=self.validation_generator,  
            validation_steps=self.validation_steps,  
            epochs=self.cfg["epochs"]  # Number of epochs  
        )  
        
    def evaluate_model(self):  
        predictions = self.model.predict(self.test_generator)    
        predicted_classes = np.argmax(predictions, axis=1)    
          
        true_classes = self.test_generator.classes    
        class_labels = list(self.test_generator.class_indices.keys())    
          
        cm = confusion_matrix(true_classes, predicted_classes)    
          
        accuracies = cm.diagonal() / cm.sum(axis=1)    
        for label, accuracy in zip(class_labels, accuracies):    
            print(f'Accuracy for class {label}: {accuracy*100:.2f}%')  
            
    def evaluate_best_treshold(self):
        predictions = self.model.predict(self.test_generator)  
        true_classes = self.test_generator.classes  
        n_classes = len(self.test_generator.class_indices)  
  
        true_one_hot = np.eye(n_classes)[true_classes]  
  
        best_threshold = 0  
        best_score = 0  
        thresholds = np.linspace(0.1, 0.9, 9)  
        f1_scores = []  
  
        for threshold in thresholds:  
            predicted_one_hot = (predictions > threshold).astype(int)  
            score = f1_score(true_one_hot, predicted_one_hot, average='macro')  
            f1_scores.append(score)  
            if score > best_score:  
                best_score = score  
                best_threshold = threshold  
  
        predicted_one_hot = (predictions > best_threshold).astype(int)  
        cm = confusion_matrix(np.argmax(true_one_hot, axis=1), np.argmax(predicted_one_hot, axis=1))  
        accuracies = cm.diagonal() / cm.sum(axis=1)  
        class_labels = list(self.test_generator.class_indices.keys())  
  
        # Plotting F1 scores  
        plt.figure(figsize=(10, 5))  
        plt.plot(thresholds, f1_scores, marker='o')  
        plt.title("F1 Score vs Threshold")  
        plt.xlabel("Threshold")  
        plt.ylabel("F1 Score")  
        plt.grid(True)  
        plt.show()  
  
        # Print and plot accuracy per class  
        plt.figure(figsize=(10, 5))  
        bars = plt.bar(class_labels, accuracies * 100)  
        plt.title("Accuracy per Class at Best Threshold")  
        plt.xlabel("Class")  
        plt.ylabel("Accuracy (%)")  
        plt.ylim([0, 100])  
  
        for bar, accuracy in zip(bars, accuracies):  
            yval = bar.get_height()  
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{accuracy*100:.2f}%', va='bottom')  # va: vertical alignment  
  
        plt.show()  
  
        print(f'Best threshold: {best_threshold:.2f} with F1 Score: {best_score:.4f}')  
        self.best_threshold = best_threshold  
        
    def inference(self, image_path, threshold=0.2):  
        # Loading and preprocessing image      
        img = image.load_img(image_path, target_size=(self.cfg["img_width"], self.cfg["img_height"]))      
        img_array = image.img_to_array(img)      
        img_array = np.expand_dims(img_array, axis=0)  # Create a batch      
        img_array /= 255.0  # Rescale image values to [0, 1]      
          
        # Make prediction      
        predictions = self.model.predict(img_array)      
        predicted_class_index = 0 if predictions[0][0] >= threshold else np.argmax(predictions[0][1:]) + 1  
        prediction_score = predictions[0][predicted_class_index]  # Get the prediction score    
  
        # Determine the class and scores for display  
        class_prediction = self.classes[predicted_class_index]  
        scores = predictions[0] * 100  # Convert scores to percentage  
        formatted_scores = ', '.join(f'{cls}({scr:.2f}%)' for cls, scr in zip(self.classes, scores))  
          
        plt.figure(figsize=(2, 2))      
          
        # Displaying the image      
        plt.imshow(img)      
        plt.axis('off')  # Don't show axes for images      
        plt.title(f'{class_prediction} | {formatted_scores}')      
        plt.show()  