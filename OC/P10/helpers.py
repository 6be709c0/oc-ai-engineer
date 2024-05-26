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

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dot, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error

import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tqdm import tqdm

import tensorflow.keras.backend as K
import numpy as np
from sklearn.metrics import roc_auc_score

from tensorflow.keras import layers, models, regularizers
from sklearn.model_selection import train_test_split

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

def preprocessing_articles(df_articles):
    df_articles['created_at_dt'] = pd.to_datetime(df_articles['created_at_ts'], unit='ms')
    df_articles.drop(columns=['created_at_ts'], inplace=True)
    
    return df_articles

def preprocessing_clicks(df_clicks):
    df_clicks['session_start_dt'] = pd.to_datetime(df_clicks['session_start'], unit='ms')
    df_clicks['click_timestamp_dt'] = pd.to_datetime(df_clicks['click_timestamp'], unit='ms')

    # Drop original timestamp columns if no longer needed
    df_clicks.drop(columns=['session_start', 'click_timestamp'], inplace=True)

    # 3. Extract additional time features
    df_clicks['click_hour'] = df_clicks['click_timestamp_dt'].dt.hour
    df_clicks['click_dayofweek'] = df_clicks['click_timestamp_dt'].dt.dayofweek
    
    return df_clicks

import tensorflow.keras.backend as K
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import numpy as np

def mean_mrr(y_true, y_pred):
    def compute_mrr(y_true, y_pred):
        y_true_np = y_true.numpy()
        y_pred_np = y_pred.numpy()
        if len(np.unique(y_true_np)) < 2:
            return 0.0
        order = np.argsort(y_pred_np)[::-1]
        y_true_sorted = np.take(y_true_np, order)
        rr = [1.0 / (i + 1) for i, x in enumerate(y_true_sorted) if x == 1]
        return np.mean(rr) if rr else 0.0
    return tf.py_function(compute_mrr, (y_true, y_pred), tf.double)


def ndcg_at_k(y_true, y_pred, k=5):
    def compute_dcg(y_true, y_pred, k):
        order = np.argsort(y_pred)[::-1]
        y_true = np.take(y_true, order[:k])
        gains = 2 ** y_true - 1
        discounts = np.log2(np.arange(1, len(y_true) + 1) + 1)
        return np.sum(gains / discounts)

    def compute_ndcg(y_true, y_pred, k):
        y_true_np = y_true.numpy()
        y_pred_np = y_pred.numpy()
        if len(np.unique(y_true_np)) < 2:
            return 0.0
        dcg = compute_dcg(y_true_np, y_pred_np, k)
        idcg = compute_dcg(y_true_np, y_true_np, k)
        return dcg / idcg if idcg > 0 else 0.0

    return tf.py_function(compute_ndcg, (y_true, y_pred, k), tf.double)

def ndcg_5(y_true, y_pred):
    return ndcg_at_k(y_true, y_pred, k=5)

def ndcg_10(y_true, y_pred):
    return ndcg_at_k(y_true, y_pred, k=10)

def g_auc(y_true, y_pred):
    def compute_auc(y_true, y_pred):
        y_true_np = y_true.numpy()
        y_pred_np = y_pred.numpy()
        if len(np.unique(y_true_np)) < 2:
            return 0.0  # Explicitly return 0.0 or another appropriate value
        try:
            return roc_auc_score(y_true_np, y_pred_np)
        except ValueError:
            return 0.0
    return tf.py_function(compute_auc, (y_true, y_pred), tf.double)

def precision_at_k(y_true, y_pred, k):
    top_k_indices = np.argpartition(y_pred, -k)[-k:]
    top_k_hits = y_true[top_k_indices]
    return np.sum(top_k_hits) / k

def recall_at_k(y_true, y_pred, k):
    top_k_indices = np.argpartition(y_pred, -k)[-k:]
    top_k_hits = y_true[top_k_indices]
    return np.sum(top_k_hits) / np.sum(y_true)
