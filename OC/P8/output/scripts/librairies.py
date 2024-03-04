# Standard library imports for system operations, file processing, etc.  
import json  
import os  
import shutil  
import subprocess  
from glob import glob  
  
# Data handling, numerical processing, and general-purpose machine learning libraries  
import numpy as np  
import pandas as pd  
from sklearn.metrics import confusion_matrix  
from sklearn.model_selection import train_test_split  
  
# Machine Learning and Neural Network frameworks  
import tensorflow as tf  
assert tf.__version__.startswith('2.'), "This script requires TensorFlow 2.x"  
from tensorflow.keras import layers, models, utils, backend as K  
from tensorflow.keras.applications import VGG16  
# Importing specific layers directly from keras.layers to maintain clarity  
from keras.layers import (BatchNormalization, Conv2D, Conv2DTranspose, Dropout, Input,   
                          Lambda, MaxPooling2D, UpSampling2D, concatenate)

from keras.models import Model


# Transformers library for models and utilities  
from transformers import (AutoFeatureExtractor, Mask2FormerForUniversalSegmentation,  
                          TrainingArguments, Trainer, pipeline, AutoImageProcessor,  
                          SegformerFeatureExtractor, SegformerForSemanticSegmentation,  
                          create_optimizer, TFAutoModelForSemanticSegmentation,  
                          DefaultDataCollator, keras_callbacks)  

import evaluate

# Image processing libraries  
from PIL import Image  
import cv2  
import torchvision.transforms.functional as TF  
import albumentations as A  
from albumentations import HorizontalFlip, VerticalFlip, Rotate  
  
# Visualization libraries  
import matplotlib.pyplot as plt  
import matplotlib as mpl  
import seaborn as sns  
  
# Parallel processing and progress bar utilities  
from pandarallel import pandarallel  
pandarallel.initialize()  
from tqdm import tqdm  
  
# Utilities for interacting with the Hugging Face Hub and hyperparameter tuning  
from huggingface_hub import cached_download, hf_hub_url  
from hyperopt import hp, STATUS_OK, fmin, tpe, Trials  
  
# IPython utilities for better notebook displays  
from IPython.display import display, HTML  
  
# MLflow for experiment tracking  
import mlflow  
import mlflow.keras  
from mlflow.models import infer_signature  
  
# Loading datasets from Hugging Face datasets library  
from datasets import load_dataset