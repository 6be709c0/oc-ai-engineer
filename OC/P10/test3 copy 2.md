Optimize this code so it can work efficiently on GPU

```python
from helpers import load_dataset, preprocessing_clicks

import pandas as pd
import numpy as np
from recommenders.datasets.python_splitters import python_stratified_split
from recommenders.models.deeprec.models.graphrec.lightgcn import LightGCN
from recommenders.models.deeprec.DataModel.ImplicitCF import ImplicitCF
from recommenders.evaluation.python_evaluation import map, ndcg_at_k, precision_at_k, recall_at_k
from recommenders.utils.constants import SEED as DEFAULT_SEED
from recommenders.utils.timer import Timer
from recommenders.utils.notebook_utils import store_metadata
from recommenders.models.deeprec.deeprec_utils import prepare_hparams
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
TOP_K = 10
EPOCHS = 50
BATCH_SIZE = 1024
SEED = DEFAULT_SEED
# Load datasets
df_articles, df_clicks, article_embeddings = load_dataset()

# Preprocessing articles
df_clicks = preprocessing_clicks(df_clicks)
# Split clicks dataset into training and test sets
df_clicks = df_clicks.sample(n=100000, random_state=SEED)
df_clicks.rename(columns={'user_id': 'userID', "click_article_id":"itemID"}, inplace=True)
df_clicks["itemID"] = df_clicks["itemID"].astype('int64')
df_clicks['rating'] = 1.0

train, test = python_stratified_split(df_clicks, ratio=0.8, col_user="userID", col_item="itemID", seed=SEED)
train.head()
train['old_index'] = train.index
train.reset_index(drop=True, inplace=True)
test['old_index'] = test.index
test.reset_index(drop=True, inplace=True)
# data.test[:5]
test[["userID", "itemID","old_index"]][:5]
data = ImplicitCF(train=train, test=test, seed=SEED, col_user="userID", col_item="itemID")
# Create userID and itemID mappings for train
user_mapping_train = dict(zip(train['userID'], data.train['userID']))
item_reverse_mapping_train = dict(zip(data.train['itemID'], train['itemID']))

# Create userID and itemID mappings for test
user_mapping_test = dict(zip(test['userID'], data.test['userID']))
item_reverse_mapping_test = dict(zip(data.test['itemID'], test['itemID']))

# Combine userID mappings
user_mapping = {**user_mapping_train, **user_mapping_test}
# Combine itemID mappings (we reverse both train and test item mappings)
item_reverse_mapping = {**item_reverse_mapping_train, **item_reverse_mapping_test}

# Example usage
user_id_example = 10
item_id_example = 0

print(f"UserID {user_id_example} corresponds to UserID {user_mapping[user_id_example]}")
print(f"ItemID {item_id_example} corresponds to ItemID {item_reverse_mapping[item_id_example]}")

user_id_example = 2151
item_id_example = 2559

print(f"UserID {user_id_example} corresponds to UserID {user_mapping[user_id_example]}")
print(f"ItemID {item_id_example} corresponds to ItemID {item_reverse_mapping[item_id_example]}")
yaml_file = "input/lightgcn.yaml"
# Hyperparameters configuration
hparams = prepare_hparams(
    yaml_file,
    n_layers=3,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    learning_rate=0.005,
    eval_epoch=5,
    top_k=TOP_K,
    save_model=True,
    save_epoch=50,
    MODEL_DIR="./input/models/"
)
model = LightGCN(hparams, data, seed=SEED)
# Train the model
with Timer() as train_time:
    model.fit()
print("Took {} seconds for training.".format(train_time.interval))
unique_item_counts = df_clicks['itemID'].nunique()
unique_item_counts
topk_scores = model.recommend_k_items(df_clicks, top_k=5, remove_seen=False)
topk_scores.head()
topk_scores[topk_scores["userID"] == 19573]
topk_scores["itemID"].dtype
# Evaluate the model
eval_map = map(test, topk_scores, k=TOP_K)
eval_ndcg = ndcg_at_k(test, topk_scores, k=TOP_K)
eval_precision = precision_at_k(test, topk_scores, k=TOP_K)
eval_recall = recall_at_k(test, topk_scores, k=TOP_K)

print("MAP:\t%f" % eval_map,
      "NDCG:\t%f" % eval_ndcg,
      "Precision@K:\t%f" % eval_precision,
      "Recall@K:\t%f" % eval_recall, sep='\n')

# print(f"UserID {user_id_example} corresponds to UserID {user_mapping[user_id_example]}")
# print(f"ItemID {item_id_example} corresponds to ItemID {item_reverse_mapping[item_id_example]}")
def get_user_article_scores(originUserID, model, data):
    userID = user_mapping[originUserID]
    try:
        # Check if the user exists in the dataset
        if userID not in data.train['userID'].values and userID not in data.test['userID'].values:
            print(f"User ID {userID} not found in the training or testing set.")
            return pd.DataFrame(columns=['user_id', 'article_id', 'score'])
        
        # Prepare a DataFrame for the specific user to get recommendations
        user_df = pd.DataFrame({'userID': [userID] * data.n_items, 'itemID': range(data.n_items)})
        
        # Use the model to score all items for the user
        full_scores = model.recommend_k_items(user_df, top_k=50, remove_seen=False)
        full_scores["userID"] = originUserID
        full_scores['prediction'] = (full_scores['prediction'] - full_scores['prediction'].min()) / (full_scores['prediction'].max() - full_scores['prediction'].min())

        # # Extract item IDs and their scores
        full_scores.rename(columns={'userID': 'user_id', "itemID":"article_id", "prediction":"score"}, inplace=True)

        return full_scores
        
    except Exception as e:
        # print(f"An error occurred: {e}")
        return pd.DataFrame(columns=['user_id', 'article_id', 'score'])
train.value_counts()
train[train["userID"] == 55694]
data.train[data.train["userID"] == 20320]
# full_scores[full_scores["article_id"] == 31836]
userID = 163
full_scores = get_user_article_scores(userID, model, data)
full_scores
full_scores
data.train
userIDs = df_clicks['userID'].unique()
userIDs.shape
from tqdm import tqdm

result_list = []
for userID in tqdm(userIDs):
    scores = get_user_article_scores(userID, model, data)
    result_list.append(scores)

result_df = pd.concat(result_list, ignore_index=True)
result_df
result_df.to_csv('result_df.csv', index=False)
result_df["itemID"].isna().sum()

```

helpers.py
```
import pickle
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor

import progressbar
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from surprise import Dataset, Reader, SVD, accuracy

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
from tensorflow.keras.layers import Embedding, Input, Lambda

from tensorflow.keras.regularizers import l2
from sklearn.metrics import roc_auc_score, ndcg_score

import pandas as pd

# Read all csv from clicks folder    
def read_csv(file_path):
    return pd.read_csv(file_path)


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
```