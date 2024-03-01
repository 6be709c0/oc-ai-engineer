# Standard library imports  
import json  # For handling JSON data  
import os  # For OS-dependent functionality  
from glob import glob  # For Unix style pathname pattern expansion  
import subprocess  
import shutil  

import mlflow  
import mlflow.keras
from mlflow.models import infer_signature

# Data handling and numerical processing  
import numpy as np  # For numerical operations  
import pandas as pd  # For data manipulation and analysis  
from sklearn.metrics import confusion_matrix  # For evaluating the accuracy of a classification  
from sklearn.model_selection import train_test_split  # For splitting data arrays into training and testing subsets  
import albumentations as A  
from albumentations import HorizontalFlip, VerticalFlip, Rotate  

# Machine Learning and Neural Network frameworks  
import tensorflow as tf  # For numerical computation using data flow graphs  
from keras.layers import (BatchNormalization, Conv2D, Conv2DTranspose, Dropout, Input,  
                          Lambda, MaxPooling2D, UpSampling2D, concatenate)  
from tensorflow.keras import backend as K  
from tensorflow.keras.layers.experimental.preprocessing import (  
    RandomFlip,  
    RandomRotation,  
    RandomZoom,  
    RandomTranslation,  
    RandomContrast,  
    RandomCrop,  
    Resizing  
)  
from keras.models import Model  # Keras is used for building and training neural networks  
from tensorflow.keras.utils import Sequence  # For generating batches of tensor image data  
from tensorflow.keras.applications.vgg16 import VGG16
from transformers import AutoFeatureExtractor, Mask2FormerForUniversalSegmentation, TrainingArguments, Trainer  
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from transformers import pipeline
from transformers import AutoImageProcessor
import evaluate
from transformers import create_optimizer
from transformers import TFAutoModelForSemanticSegmentation
from transformers import DefaultDataCollator
from transformers.keras_callbacks import KerasMetricCallback
# Image processing  
from PIL import Image  # For opening, manipulating, and saving many different image file formats  
import cv2  # For computer vision tasks  
import torchvision.transforms.functional as TF  # For transforming PIL images into tensors  
  
# Visualization  
import matplotlib as mpl  
import matplotlib.pyplot as plt  # For creating static, interactive, and animated visualizations in Python  
import seaborn as sns  # For making statistical graphics in Python  
  
# Parallel processing  
from pandarallel import pandarallel  # For parallel data processing  
pandarallel.initialize()

# Progress bar  
from tqdm import tqdm  # For adding progress meters  
from huggingface_hub import cached_download, hf_hub_url
from hyperopt import hp, STATUS_OK, fmin, tpe, Trials  

# IPython display for notebooks  
from IPython.display import display, HTML  # For embedding rich web content in IPython notebooks  
from datasets import load_dataset
# Ensuring TensorFlow 2.x is used  
assert tf.__version__.startswith('2.')  
