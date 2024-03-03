base_config = {
    
    # Width of the image used for training
    "width": 512, 
    # Height of the image used for training
    "height": 256,
    
    # Number of classes 
    "classes": 8, 
    
    # Use image augmentation
    "use_augment": True, 
    # Create x image per augmentation.
    # So If set to two, when processing an image, it will create x augmented version of the same image
    "image_per_augment": 1,
    
    # Number of loop for checking hyperparameter
    "max_evals": 1,
    
    # Mlflow server
    "mlwflow_tracking_uri": "http://127.0.0.1:5000",
    # Experiment title of MLFlow
    "mlwflow_experiment_title": "",
    
    # Number of sample to use for training
    "train_sample_nb": 0,
    # Number of sample to use for validation
    "val_sample_nb": 0,
    
    # Reuse a saved model
    "use_saved_model_path": "",
    
    # City scape dataset path
    "gtFine_path": "./data/P8_Cityscapes_gtFine_trainvaltest.zip", 
    "leftImg8bit_path": "./data/P8_Cityscapes_leftImg8bit_trainvaltest.zip",
    "data_path": "./data/gtFine",
}
