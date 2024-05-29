from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import progressbar

import os

from surprise import Reader, Dataset
from surprise.model_selection import train_test_split, cross_validate

from sklearn.metrics.pairwise import linear_kernel

# Read all csv from clicks folder    
def read_csv(file_path):
    return pd.read_csv(file_path)


def load_dataset():
    # Load datasets
    df_articles = pd.read_csv('input/archive/articles_metadata.csv')
    df_clicks_sample = pd.read_csv('input/archive/clicks_sample.csv')
    folder_path = 'input/archive/clicks/clicks'

    csv_files_clicks = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]

    with ProcessPoolExecutor() as executor:
        df_clicks = list(executor.map(read_csv, csv_files_clicks))

    df_clicks = pd.concat(df_clicks)
    df_clicks.reset_index(drop=True, inplace=True)
    
    return df_articles, df_clicks
