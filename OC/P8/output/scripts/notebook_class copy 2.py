from librairies import *
from download_dataset import *
from functions import *
from data_generator import DataGenerator

class NotebookProcessor:
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
    
    category_mapping_polygons = {    
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
    
    def __init__(self, config):  
        self.cfg = config  
        self.prepare_notebook()
        
        self.category_ids = {cat: idx for idx, cat in enumerate(self.categories.keys())} 
  
    def print_config(self):  
        display(HTML('<h1 style="padding:0;margin:0;"><b>CONFIG USED</b></h1>'))    
        print(json.dumps(self.cfg, indent=4), "\n")  
  
    def set_model(self, model):  
        self.model = model
        
        # model.compile(optimizer="adam", loss=['categorical_crossentropy'], metrics=["accuracy"])
        model.compile(optimizer="adam", loss=['categorical_crossentropy'], metrics=[dice_coef, iou, "accuracy"])
        model.summary()

    def set_dataframes(self):  
        json_paths_train = glob(f'{self.cfg["data_path"]}/train/**/*.json')
        json_paths_test = glob(f'{self.cfg["data_path"]}/test/**/*.json')
        json_paths_val = glob(f'{self.cfg["data_path"]}/val/**/*.json')

        data_train = [get_file_info(path) for path in json_paths_train]
        data_test = [get_file_info(path) for path in json_paths_test]
        data_val = [get_file_info(path) for path in json_paths_val]

        columns = ["original_img_path", "color_img_path", "instance_ids_img_path", "label_ids_img_path", "polygons_json_path"]
        df_train = pd.DataFrame(data_train, columns=columns)
        df_test = pd.DataFrame(data_test, columns=columns)
        df_val = pd.DataFrame(data_val, columns=columns)
        
        if self.cfg["train_sample_nb"] > 0:
            print(f"\n- Sampling the training dataset from {df_train.shape[0]} to {self.cfg['train_sample_nb']}.")
            df_train = df_train.sample(n=self.cfg["train_sample_nb"], random_state=42).reset_index(drop=True)
            # df_test = df_test.sample(n=sample_dataset, random_state=42).reset_index(drop=True)
            # df_val = df_val.sample(n=100, random_state=42).reset_index(drop=True)
        else:
            print("- Using the full training dataset")
            
        if self.cfg["val_sample_nb"] > 0:
            print(f"\n- Sampling the validation dataset from {df_val.shape[0]} to {self.cfg['val_sample_nb']}.")
            df_val = df_val.sample(n=self.cfg["val_sample_nb"], random_state=42).reset_index(drop=True)
        else:
            print("- Using the full validation dataset")
        
        self.dfs = {
            "train": df_train,
            "test": df_test,
            "val": df_val,
        }
        
        self.img = {
            "train": df_train["original_img_path"].tolist(),
            "test": df_test["original_img_path"].tolist(),
            "val": df_val["original_img_path"].tolist(),
        }
        
        self.mask = {
            "train": df_train["label_ids_img_path"].tolist(),
            "test": df_test["label_ids_img_path"].tolist(),
            "val": df_val["label_ids_img_path"].tolist(),
        }
        
        self.mask_polygons = {
            "train": df_train["polygons_json_path"].tolist(),
            "test": df_test["polygons_json_path"].tolist(),
            "val": df_val["polygons_json_path"].tolist(),
        }
        
        self.tf_ds = {
            "train": self.tf_dataset(self.img["train"], self.mask["train"], self.cfg["batch_size"]),
            "val": self.tf_dataset(self.img["val"], self.mask["val"], self.cfg["batch_size"]),
        }
        
        self.train_generator = DataGenerator(self.img['train'], self.mask['train'], self.cfg['batch_size'], self.cfg, shuffle=True)  
        self.val_generator = DataGenerator(self.img['val'], self.mask['val'], self.cfg['batch_size'], self.cfg, shuffle=False)  
        
    def model_save(self, path):
        self.model.save(path)
        
    def model_fit(self):
        
        if(self.cfg["use_saved_model_path"]):
            print("Skipping because of use_saved_model_path set in config")
            print(f"Loading model from config {self.cfg['use_saved_model_path']}")
            
            model = tf.keras.models.load_model(self.cfg["use_saved_model_path"]) 
            self.set_model(model)
            
            return
    
        early_stopping = tf.keras.callbacks.EarlyStopping(  
            monitor='val_loss',   
            min_delta=0.001,   
            patience=10,   
            verbose=1,   
            restore_best_weights=True  
        )

        train_steps = len(self.img["train"])//self.cfg["batch_size"]
        valid_steps = len(self.img["val"])//self.cfg["batch_size"]
        
        print(f"\nTrain steps: {train_steps}")
        print(f"Balidation steps: {valid_steps}")
        print("\n---------------------\n\n")
        
        # history = self.model.fit(self.tf_ds["train"],
        #     steps_per_epoch=train_steps,
        #     validation_data=self.tf_ds["val"],
        #     validation_steps=valid_steps,
        #     epochs=self.cfg["epoch"],
        #     callbacks=[early_stopping]
        # )
        history = self.model.fit(  
            self.train_generator,  
            validation_data=self.val_generator,  
            epochs=self.cfg["epoch"],  
            callbacks=[early_stopping]  # Assuming early_stopping callback is defined  
        )  
        self.model_fit_history = history
        print("\nModel trained!\n")
        
    def read_image(self, x):  
        x = cv2.imread(x, cv2.IMREAD_COLOR)  
        x = cv2.resize(x, (self.cfg["width"], self.cfg["height"]))  
        x = x/255.0  
        x = x.astype(np.float32)  
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        return x  
    
    def read_mask(self, x):  
        original_mask = cv2.imread(x, cv2.IMREAD_GRAYSCALE)   
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
    
    def read_mask_polygons(self, path):  
        mask_img_array = polygon_to_mask_img(path, self.category_mapping_polygons)
        color_map = mpl.colormaps.get_cmap('viridis')  
        
        mask_img = cv2.resize(mask_img_array, (self.cfg["width"], self.cfg["height"]))  
        return mask_img.astype(np.int32)
    
    
    def preprocess_test(self, x):
        def f(x):
            x = x.decode()
            image = self.read_image(x)
            return image
        
        image = tf.convert_to_tensor(tf.numpy_function(f, [x] , [tf.float32]))
        image = tf.reshape(image, (self.cfg["height"], self.cfg["width"], 3))  
        return image
        
    def preprocess(self, x,y):
        def f(x,y):
            x = x.decode() ##byte stream conversion
            y = y.decode()
            image = self.read_image(x)
            mask = self.read_mask(y)
            return image, mask
        
        image, mask = tf.numpy_function(f,[x,y],[tf.float32, tf.int32])
        mask = tf.one_hot(mask, self.cfg["classes"], dtype=tf.int32)
        
        image.set_shape([self.cfg["height"], self.cfg["width"], 3])    
        mask.set_shape([self.cfg["height"], self.cfg["width"], self.cfg["classes"]])
        # mask.set_shape([self.cfg["height"], self.cfg["width"], self.cfg["classes"]])
        
        return image, mask
    
    def tf_dataset(self, x,y, batch=4):
        dataset = tf.data.Dataset.from_tensor_slices((x,y)) # Dataset data object created from input and target data
        dataset = dataset.shuffle(buffer_size=100) ## selected from the first 100 samples
        dataset = dataset.map(self.preprocess) # Applying preprocessing to every batch in the Dataset object
        
        dataset = dataset.batch(batch) # Determine batch-size
        dataset = dataset.repeat()
        dataset = dataset.prefetch(2) # Optimization to reduce waiting time on each object
        return dataset
  
    def test_dataset(self, x, batch=32):
        dataset = tf.data.Dataset.from_tensor_slices(x)
        dataset = dataset.map(self.preprocess_test)
        dataset = dataset.batch(batch)
        dataset = dataset.prefetch(2)
        return dataset
    
    def prepare_notebook(self):  
        self.print_config()  
        download_dataset(self.cfg)  
        unzip_dataset(self.cfg)
        
        self.set_dataframes()
          
    def read_overlay_pred(self, image, mask, mask_opacity=0.5):
        color_map = mpl.colormaps.get_cmap('viridis')  

        mask_img_array_normalized = mask / mask.max()  
        mask_img_colored = color_map(mask_img_array_normalized)[:, :, :3]  # Exclude the alpha channel  
        mask_img_colored = (mask_img_colored * 255).astype(np.uint8)  
        
        print("AA", image.shape)
        print("BB", mask_img_colored.shape)
        
        overlayed = cv2.addWeighted(src1=image, alpha=(1-mask_opacity), src2=mask_img_colored.astype(np.float32)/255.0, beta=mask_opacity, gamma=0)  
        
        return overlayed
    
    def model_predict_with_display_and_accuracy(self, image_path, truth_mask_path):  
        image = self.read_image(image_path)  
        image_to_predict = np.expand_dims(image, axis=0)  # Add batch dimension  

        ground_truth_mask = self.read_mask(truth_mask_path)
        ground_truth_mask = cv2.resize(ground_truth_mask, (self.model.input_shape[2], self.model.input_shape[1]), interpolation=cv2.INTER_NEAREST)  # Resizing to match the model's input  

        # Predict the mask using the model  
        prediction = self.model.predict(image_to_predict)
        predicted_mask = tf.argmax(prediction, axis=-1)[0]
        predicted_mask_np = predicted_mask.cpu().numpy()
        correct_predictions_mask = np.where(predicted_mask == ground_truth_mask, predicted_mask, np.nan)  # Use np.nan for non-correct pixels    
        correct_predictions_image = np.where(np.expand_dims(predicted_mask_np == ground_truth_mask, axis=-1), image, 1) # This replaces incorrect predictions with black  
        accuracy = np.mean(predicted_mask == ground_truth_mask)  # Assumes truth_mask is already an int32 numpy array of labels  
        
        show_grid_prediction(
            image, 
            ground_truth_mask, 
            predicted_mask_np,
            correct_predictions_mask,
            correct_predictions_image,
            accuracy
        )
         
    
    def model_predict_with_display_and_accurac_deprecated(self, image_path, truth_mask_path):  
        # Load and preprocess the image  
        image = self.read_image(image_path)
        image_to_predict = np.expand_dims(image, axis=0)  # Add batch dimension  
        
        # Load and preprocess the truth mask  
        truth_mask = self.read_mask(truth_mask_path)  # Using the read mask function defined earlier  
        truth_mask_resized = cv2.resize(truth_mask, (self.model.input_shape[2], self.model.input_shape[1]), interpolation=cv2.INTER_NEAREST)  # Resizing to match the model's input  
    
        # Predict the mask using the model  
        prediction = self.model.predict(image_to_predict)  
        predicted_mask = tf.argmax(prediction, axis=-1)[0]  # Remove batch dimension and convert prediction to class labels  
    
        # Calculate accuracy  
        accuracy = np.mean(predicted_mask == truth_mask_resized)  # Assumes truth_mask is already an int32 numpy array of labels  
        
        overlay_img = self.read_overlay_pred(image, predicted_mask.numpy())
        
        correct_predictions_mask = np.where(predicted_mask == truth_mask_resized, predicted_mask, np.nan)  # Use np.nan for non-correct pixels    
        correct_predictions_image = np.where(np.expand_dims(predicted_mask.numpy() == truth_mask_resized, axis=-1), image, 1) # This replaces incorrect predictions with black  

        # Display images
        titles_and_images = [  
            ('Original Image', image),  
            ('Ground Truth Mask', truth_mask_resized),  
            ('Predicted Mask', predicted_mask),  
            ('Predicted Mask Overlay', overlay_img),
            ('Correct Predictions Mask', correct_predictions_mask),
            ('Original image cropped', correct_predictions_image),
        ]  
        
        # Setting up the figure for 2 rows and 2 columns  
        plt.figure(figsize=(16, 5))  
        plt.suptitle(f'Accuracy {accuracy*100:.2f}%', fontsize=16, fontweight='bold', y=1.01)  
        
        for i, (title, img) in enumerate(titles_and_images, start=1):
            plt.subplot(2, 3, i)  
            plt.title(title)
            plt.imshow(img)
                
            plt.axis('off') 
            
        plt.show()  
    
        print(f"Accuracy: {accuracy*100:.2f}%") 
         
    def display_mask_and_accuracy(self, image_path, truth_mask_path, predicted_mask):  
        # Load and preprocess the image  
        image = self.read_image(image_path)
        image_to_predict = np.expand_dims(image, axis=0)  # Add batch dimension  
        
        # Load and preprocess the truth mask  
        truth_mask = self.read_mask(truth_mask_path)  # Using the read mask function defined earlier  
        truth_mask_resized = cv2.resize(truth_mask, (self.cfg["height"], self.cfg["width"]), interpolation=cv2.INTER_NEAREST)  # Resizing to match the model's input  

        # # Predict the mask using the model  
        # prediction = self.model.predict(image_to_predict)  
        # predicted_mask = tf.argmax(prediction, axis=-1)[0]  # Remove batch dimension and convert prediction to class labels  
    
        # Calculate accuracy  
        accuracy = np.mean(predicted_mask == truth_mask_resized)  # Assumes truth_mask is already an int32 numpy array of labels  
        
        overlay_img = self.read_overlay_pred(image, predicted_mask.numpy())
        
        correct_predictions_mask = np.where(predicted_mask == truth_mask_resized, predicted_mask, np.nan)  # Use np.nan for non-correct pixels    
        
        correct_predictions_image = np.where(np.expand_dims(predicted_mask.numpy() == truth_mask_resized, axis=-1), image, 1) # This replaces incorrect predictions with black  

        # Display images
        titles_and_images = [  
            ('Original Image', image),  
            ('Ground Truth Mask', truth_mask_resized),  
            ('Predicted Mask', predicted_mask),  
            ('Predicted Mask Overlay', overlay_img),
            ('Correct Predictions Mask', correct_predictions_mask),
            ('Original image cropped', correct_predictions_image),
        ]  
        
        # Setting up the figure for 2 rows and 2 columns  
        plt.figure(figsize=(16, 5))  
        plt.suptitle(f'Accuracy {accuracy*100:.2f}%', fontsize=16, fontweight='bold', y=1.01)  
        
        for i, (title, img) in enumerate(titles_and_images, start=1):
            plt.subplot(2, 3, i)  
            plt.title(title)
            plt.imshow(img)
                
            plt.axis('off') 
            
        plt.show()  
    
        print(f"Accuracy: {accuracy*100:.2f}%")  
        
        
    def model_predict_with_display(self, image_path):  
        # Load and preprocess the image  
        image = self.read_image(image_path)  
        image_to_predict = np.expand_dims(image, axis=0)  # Add batch dimension  
        
        # Predict the mask using the model  
        prediction = self.model.predict(image_to_predict)  
        predicted_mask = tf.argmax(prediction, axis=-1)[0]  # Remove batch dimension and convert prediction to class labels  
        overlay_img = self.read_overlay_pred(image, predicted_mask.numpy())
        # Display the original image, the predicted mask, and print accuracy  
        plt.figure(figsize=(16, 6))  
        
        plt.subplot(1, 3, 1)  
        plt.title('Original Image')  
        plt.imshow(image)  
        plt.axis('off')  
        
        plt.subplot(1, 3, 2)  
        plt.title('Predicted Mask')  
        plt.imshow(predicted_mask, cmap='viridis')  
        plt.axis('off')  
    
        plt.subplot(1, 3, 3)  
        plt.title('Original + Pred Mask')  
        plt.imshow(overlay_img, cmap='viridis')  
        plt.axis('off')  
        
        plt.show()