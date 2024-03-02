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
    
    def __init__(self, config):  
        self.cfg = config  
        self.prepare_notebook()
        
        self.category_ids = {cat: idx for idx, cat in enumerate(self.categories.keys())}
        
        if self.cfg["mlwflow_tracking_uri"]:
            mlflow.set_tracking_uri(uri=self.cfg["mlwflow_tracking_uri"])
        
    def print_config(self):  
        display(HTML('<h1 style="padding:0;margin:0;"><b>CONFIG USED</b></h1>'))    
        print(json.dumps(self.cfg, indent=4), "\n")  
  
    def set_model(self, model):  
        self.model = model
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
        
        # Define an augmentation pipeline  
        self.augmentation_pipeline = A.Compose([      
            A.HorizontalFlip(p=0.5),  
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, border_mode=0, p=0.5),
            A.RandomResizedCrop(height=self.cfg["height"], width=self.cfg["width"], scale=(0.8, 1.0), ratio=(0.75, 1.33), p=0.5),
        ])
        
        self.test_generator = DataGenerator(self.img['test'], self.mask['test'], 3, self.cfg, shuffle=False)  
        
        if self.cfg["use_augment"]:
            self.sample_generator = DataGenerator(self.img['train'], self.mask['train'], 3, self.cfg, shuffle=True, augmentation=self.augmentation_pipeline)  
        else:
            self.sample_generator = DataGenerator(self.img['train'], self.mask['train'], 3, self.cfg, shuffle=True, augmentation=None)  
        
    def model_save(self, path):
        self.model.save(path)
        
    def objective(self, params):
        early_stopping = tf.keras.callbacks.EarlyStopping(  
            monitor='val_loss',   
            min_delta=0.001,   
            patience=10,   
            verbose=1,   
            restore_best_weights=True  
        )
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=params['learning_rate'])  
        
        self.model.compile(optimizer=optimizer, loss=['categorical_crossentropy'], metrics=[dice_coef, iou, "accuracy"])
        
        cfg = {
            **self.cfg,
            "image_per_augment": params["image_per_augment"]
        }    

        train_generator = DataGenerator(self.img['train'], self.mask['train'], params["batch_size"], cfg, shuffle=True, augmentation=self.augmentation_pipeline)  
        val_generator = DataGenerator(self.img['val'], self.mask['val'], params["batch_size"], cfg, shuffle=False)  
        
        if(self.cfg["mlwflow_tracking_uri"]):
            mlflow.set_experiment(self.cfg["mlwflow_experiment_title"])  
            with mlflow.start_run():
                
                mlflow.log_params({
                    **self.cfg,
                    **params
                })    
                
                history = self.model.fit(  
                    train_generator,  
                    validation_data=val_generator,  
                    epochs=params["epochs"],  
                    callbacks=[early_stopping]  # Assuming early_stopping callback is defined  
                ) 
                
                mlflow.log_metric("best_loss", min(history.history['loss']))
                mlflow.log_metric("best_dice_coef", min(history.history['dice_coef']))
                mlflow.log_metric("best_iou", min(history.history['iou']))
                mlflow.log_metric("best_accuracy", max(history.history['accuracy']))
                
                mlflow.log_metric("best_val_accuracy", max(history.history['val_accuracy']))
                mlflow.log_metric("best_val_loss", min(history.history['val_loss']))          
                mlflow.log_metric("best_val_dice_coef", min(history.history['val_dice_coef']))
                mlflow.log_metric("best_val_iou", min(history.history['val_iou']))
                
                self.model.save("model.keras")
                mlflow.log_artifact('model.keras')
                
                self.model_fit_history = history
        else:
            history = self.model.fit(  
                self.train_generator,  
                validation_data=self.val_generator,  
                epochs=params["epochs"],  
                callbacks=[early_stopping]  # Assuming early_stopping callback is defined  
            ) 
            self.model_fit_history = history
            
        return {'loss': -max(history.history['val_accuracy']), 'status': STATUS_OK}  
 
    def model_fit(self, space=None):
        
        if(self.cfg["use_saved_model_path"]):
            print("Skipping because of use_saved_model_path set in config")
            print(f"Loading model from config {self.cfg['use_saved_model_path']}")
            
            model = tf.keras.models.load_model(self.cfg["use_saved_model_path"]) 
            self.set_model(model)
            return
        
        if space == None:
            space = {
                'image_per_augment': hp.choice('image_per_augment', [1]),
                'batch_size': hp.choice('batch_size', [3]),
                'epochs': hp.choice('epochs', [12]),
                'learning_rate': hp.choice('learning_rate', [1e-3]),  
            }
            # space = {
            #     'image_per_augment': hp.choice('image_per_augment', [1, 2, 3, 4, 5, 6, 7, 8]),
            #     'batch_size': hp.choice('batch_size', [3, 4, 8, 16]),
            #     'epochs': hp.choice('epochs', [4, 8, 12, 24]),
            #     'learning_rate': hp.uniform('learning_rate', 0.0001, 0.001),  
            # }
        
        trials = Trials()
        fmin(  
            fn=lambda params: self.objective(params),  
            space=space,  
            algo=tpe.suggest,  
            max_evals=self.cfg["max_evals"],
            trials=trials  
        )
        
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
    
    def prepare_notebook(self):  
        self.print_config()  
        download_dataset(self.cfg)  
        unzip_dataset(self.cfg)
        move_val_test(self.cfg)
        
        self.set_dataframes()
          
    def read_overlay_pred(self, image, mask, mask_opacity=0.5):
        color_map = mpl.colormaps.get_cmap('viridis')  

        mask_img_array_normalized = mask / mask.max()  
        mask_img_colored = color_map(mask_img_array_normalized)[:, :, :3]  # Exclude the alpha channel  
        mask_img_colored = (mask_img_colored * 255).astype(np.uint8)  
        
        overlayed = cv2.addWeighted(src1=image, alpha=(1-mask_opacity), src2=mask_img_colored.astype(np.float32)/255.0, beta=mask_opacity, gamma=0)  
        
        return overlayed
    
    def model_predict_with_display_and_accuracy(self, image_path, truth_mask_path):  
        # Reading and preprocessing the image and mask    
        image = self.read_image(image_path)  
        image_to_predict = np.expand_dims(image, axis=0)  # Adding batch dimension  
        ground_truth_mask = self.read_mask(truth_mask_path)  
        ground_truth_mask = cv2.resize(ground_truth_mask, (self.model.input_shape[2], self.model.input_shape[1]), interpolation=cv2.INTER_NEAREST)  
    
        # Prediction and calculation of correct predictions and accuracy  
        prediction = self.model.predict(image_to_predict)  
        predicted_mask = tf.argmax(prediction, axis=-1)[0]  
        predicted_mask_np = predicted_mask.numpy()  # Adjust for TensorFlow 2.x  
        correct_predictions_image = np.where((predicted_mask_np[:, :, None] == ground_truth_mask[:, :, None]), image, 1)  # Replacing incorrect predictions with black  
        correct_predictions_mask = np.where(predicted_mask == ground_truth_mask, predicted_mask, np.nan)  # Use np.nan for non-correct pixels    
        accuracy = np.mean(predicted_mask_np == ground_truth_mask)  
        
        # Calculating per-class accuracy  
        unique_classes_in_truth = np.unique(ground_truth_mask)  # Find unique classes in the ground truth mask  
        conf_matrix = confusion_matrix(ground_truth_mask.flatten(), predicted_mask_np.flatten(), labels=unique_classes_in_truth)  
        per_class_accuracy = np.nan_to_num(conf_matrix.diagonal() / conf_matrix.sum(axis=1))
        categories = ['void', 'flat', 'construction', 'object', 'nature', 'sky', 'human', 'vehicle']  
    
        # Filtering categories, accuracies, and pixel counts based on what's present in the ground truth  
        present_categories = np.array(categories)[unique_classes_in_truth]  
        sorted_indices_desc = np.argsort(per_class_accuracy)[::-1]  
        sorted_accuracies_desc = per_class_accuracy[sorted_indices_desc]  
        sorted_categories_desc = present_categories[sorted_indices_desc]  
        sorted_pixel_counts_desc = np.array([np.sum(ground_truth_mask == i) for i in unique_classes_in_truth])[sorted_indices_desc]  
    
        # Converting accuracies to percentages  
        sorted_accuracies_percent_desc = sorted_accuracies_desc * 100  
        
        # Including accuracy print for clarity  
        show_grid_prediction(
            image, 
            ground_truth_mask, 
            predicted_mask_np,
            correct_predictions_mask,
            correct_predictions_image
        )
        
        # Plotting  
        fig, axs = plt.subplots(1, 2, figsize=(20, 5))  # Adjusted figsize for potentially better layout  
        axs[0].barh(sorted_categories_desc, sorted_accuracies_percent_desc, color='skyblue', height=0.5)  # Adjusted bar width  
        axs[0].set_xlabel('Accuracy (%)', fontsize=12)  # Adjusted font size  
        axs[0].set_title('Pixel accuracy per class', fontsize=14)  # Adjusted font size  
        axs[0].tick_params(axis='both', which='major', labelsize=10)  # Adjusted tick label size  
        axs[0].grid(True, linestyle='--', linewidth=0.5)  # Added gridlines  
        
        for index, value in enumerate(sorted_accuracies_percent_desc):  
            axs[0].text(value + 1, index, f'{value:.2f}%', va='center', fontsize=10)  # Adjusted text positioning and font size  
        
        axs[1].barh(sorted_categories_desc, sorted_pixel_counts_desc, color='lightgreen', height=0.5)  # Adjusted bar width  
        axs[1].set_xlabel('Number of Pixels', fontsize=12)  # Adjusted font size  
        axs[1].set_title('Original pixel counts per class', fontsize=14)  # Adjusted font size  
        axs[1].tick_params(axis='both', which='major', labelsize=10)  # Adjusted tick label size  
        axs[1].grid(True, linestyle='--', linewidth=0.5)  # Added gridlines  
        
        for index, value in enumerate(sorted_pixel_counts_desc):  
            axs[1].text(value + 1, index, str(value), va='center', fontsize=10)  # Adjusted text positioning and font size  
        
        plt.tight_layout()  
        plt.show()  
         
    def model_inference_with_display(self, image_path):  
        # Load and preprocess the image  
        image = self.read_image(image_path)  
        image_to_predict = np.expand_dims(image, axis=0)
        
        prediction = self.model.predict(image_to_predict)  
        predicted_mask = tf.argmax(prediction, axis=-1)[0] 

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
        plt.imshow(image)
        plt.imshow(predicted_mask, cmap='viridis', alpha=0.5)  
        plt.axis('off')  
        
        plt.show()