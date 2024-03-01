base_config = {
    "width": 512,
    "height": 256,
    "classes": 8,
    "use_augment": True,
    "image_per_augment": 1,
    "max_evals": 1,
    "mlwflow_tracking_uri": "http://127.0.0.1:5000",
    "mlwflow_experiment_title": "",
    "train_sample_nb": 0,
    "val_sample_nb": 0,
    "use_saved_model_path": "",
    
    "gtFine_path": "./data/P8_Cityscapes_gtFine_trainvaltest.zip", 
    "leftImg8bit_path": "./data/P8_Cityscapes_leftImg8bit_trainvaltest.zip",
    "data_path": "./data/gtFine",
}
