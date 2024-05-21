import pickle
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor
from read_csv_module import read_csv

import progressbar
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split as train_test_split_surprise

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split as train_test_split_sklearn
from sklearn.preprocessing import normalize

from scipy.sparse.linalg import svds

import matplotlib.pyplot as plt
import seaborn as sns

def load_dataset():
    # Load datasets
    df_articles = pd.read_csv('input/archive/articles_metadata.csv')
    df_clicks_sample = pd.read_csv('input/archive/clicks_sample.csv')
    folder_path = 'input/archive/clicks/clicks'

    with open('input/archive/articles_embeddings.pickle', 'rb') as file:
        article_embeddings = pickle.load(file)
        
    csv_files_clicks = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]

    with ProcessPoolExecutor() as executor:
        df_clicks = list(executor.map(read_csv, csv_files_clicks))

    df_clicks = pd.concat(df_clicks)
    df_clicks.reset_index(drop=True, inplace=True)
    
    return df_articles, df_clicks, article_embeddings.astype(np.float32)