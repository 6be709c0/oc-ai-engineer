base_config = {
    "width": 0,
    "height": 0,
    "classes": 8,
    "epoch": 3,
    "use_augment": True,
    "augment_per_image": 1,
    "batch_size": 3,
    "mlwflow_tracking_uri": "http://127.0.0.1:5000",
    "mlwflow_experiment_title": "",
    "train_sample_nb": 100,
    "val_sample_nb": 20,
    "use_saved_model_path": "",
    
    "gtFine_path": "./data/P8_Cityscapes_gtFine_trainvaltest.zip", 
    "leftImg8bit_path": "./data/P8_Cityscapes_leftImg8bit_trainvaltest.zip",
    "data_path": "./data/gtFine",
}

base_config_resized = {
    **base_config,
    "width": 256,
    "height": 128,
}

base_config_full = {
    **base_config,
    "width": 2048,
    "height": 1024,
}