Optimize this code so it can work efficiently on GPU

```python
from helpers import *
# Load datasets
df_articles, df_clicks, article_embeddings = load_dataset()
pca = PCA()
pca.fit(article_embeddings)

# Variance data
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
components = np.arange(len(cumulative_variance)) + 1

# Plotting
plt.figure(figsize=(10,4))
plt.plot(components, cumulative_variance, label='Cumulative Explained Variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance (%)')
plt.title('PCA Explained Variance')

# Annotate specific variance percentages
variance_thresholds = [0.9, 0.95, 0.97, 0.98, 0.99]
for threshold in variance_thresholds:
    component_number = np.where(cumulative_variance >= threshold)[0][0]
    plt.scatter(component_number + 1, cumulative_variance[component_number], color='red')
    plt.annotate(f"{int(threshold*100)}%", (component_number + 1, cumulative_variance[component_number]),
                 textcoords="offset points", xytext=(0,10), ha='center')

plt.grid(True)
plt.show()
pca = PCA(n_components=0.98)
reduced_embeddings = pca.fit_transform(article_embeddings)
# reduced_embeddings = pca.fit_transform(article_embeddings)
print(reduced_embeddings.shape)
article_embeddings = reduced_embeddings
# cosine_sim_old = cosine_similarity(reduced_embeddings)
# preprocess data
df_articles = preprocessing_articles(df_articles)
df_clicks = preprocessing_clicks(df_clicks)
article_embeddings_df = pd.DataFrame(article_embeddings)
print("article_embeddings shape", article_embeddings_df.shape)
print("df_articles shape", df_articles.shape)
articles_clicked = df_clicks.click_article_id.value_counts().index
df_articles = df_articles.loc[articles_clicked]
article_embeddings_df = article_embeddings_df.loc[articles_clicked]

print("df_articles shape", df_articles.shape)
print("article_embeddings shape", article_embeddings_df.shape)
def train_test_split_sessions(clicks_df, test_size=0.1, val_size=0.1, random_state=42):
    session_ids = clicks_df['session_id'].unique()
    train_sessions, test_sessions = train_test_split(session_ids, test_size=test_size, random_state=random_state)
    train_sessions, val_sessions = train_test_split(train_sessions, test_size=val_size, random_state=random_state)
    
    train_df = clicks_df[clicks_df['session_id'].isin(train_sessions)]
    val_df = clicks_df[clicks_df['session_id'].isin(val_sessions)]
    test_df = clicks_df[clicks_df['session_id'].isin(test_sessions)]
    all_df = clicks_df[clicks_df['session_id'].isin(session_ids)]
    
    return train_df, val_df, test_df, all_df
 

# Split the clicks dataframe
train_clicks_df, val_clicks_df, test_clicks_df, all_clicks_df = train_test_split_sessions(df_clicks)

print(f"Training clicks shape: {train_clicks_df.shape}")
print(f"Validation clicks shape: {val_clicks_df.shape}")
print(f"Testing clicks shape: {test_clicks_df.shape}")
print(f"All clicks shape: {all_clicks_df.shape}")
#### Merging Articles Embeddings with Articles Metadata

# Merging with articles_metadata
# articles_merged_df = pd.merge(df_articles, article_embeddings_df, on='article_id')
tqdm.pandas()

def create_user_profiles(clicks_df, article_embeddings_df):
    user_profiles = clicks_df.groupby('user_id')['click_article_id'].apply(list).reset_index()
    embeddings_dict = article_embeddings_df.T.to_dict('list')
    
    user_profiles['user_embedding'] = user_profiles['click_article_id'].progress_apply(
        lambda x: np.mean([embeddings_dict[article] for article in x if article in embeddings_dict], axis=0)
    )
    
    return user_profiles

user_profiles_df_train = create_user_profiles(train_clicks_df, article_embeddings_df)
user_profiles_df_test = create_user_profiles(test_clicks_df, article_embeddings_df)
user_profiles_df_val = create_user_profiles(val_clicks_df, article_embeddings_df)
user_profiles_df_all = create_user_profiles(all_clicks_df, article_embeddings_df)
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

def create_content_based_model(input_dim):
    model = models.Sequential()
    # Input Layer
    model.add(layers.Input(shape=(input_dim,)))
    
    # Hidden Layers
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.2))
    
    # Output Layer - Predicting the relevance score
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[ndcg_5, ndcg_10, mean_mrr, g_auc])
    return model
# Prepare data
def prepare_data(user_profiles_df_train, articles_df, articles_embeddings_df):
    X = []
    y = []
    
    embeddings_dict = articles_embeddings_df.T.to_dict('list')
    
    for i, user in tqdm(user_profiles_df_train.iterrows(), total=len(user_profiles_df_train)):
        if i >= 500:
            break
        
        user_embedding = user['user_embedding']
        clicked_articles = user['click_article_id']
        
        for article_id in clicked_articles:
            if article_id in embeddings_dict:
                article_embedding = embeddings_dict[article_id]
                combined_features = np.concatenate((user_embedding, article_embedding))
                X.append(combined_features)
                y.append(1) # Positive sample
        
        # Add some negative samples for training
        negative_samples = articles_df[~articles_df['article_id'].isin(clicked_articles)]['article_id'].sample(n=len(clicked_articles))
        
        for article_id in negative_samples:
            if article_id in embeddings_dict:
                article_embedding = embeddings_dict[article_id]
                combined_features = np.concatenate((user_embedding, article_embedding))
                X.append(combined_features)
                y.append(0) # Negative sample
                
    X = np.array(X)
    y = np.array(y)
    
    return X, y
X_train, y_train = prepare_data(user_profiles_df_train, df_articles, article_embeddings_df)
X_val, y_val = prepare_data(user_profiles_df_val, df_articles, article_embeddings_df)
# import tensorflow.keras.backend as K
# import numpy as np
# from sklearn.metrics import roc_auc_score
# from tqdm import tqdm
# import numpy as np

# def precision_at_k(true_labels, pred_scores, k=5):
#     top_k_indices = np.argsort(pred_scores)[-k:]
#     top_k_true_labels = true_labels[top_k_indices]
#     return np.sum(top_k_true_labels) / k

# def recall_at_k(true_labels, pred_scores, k=5):
#     top_k_indices = np.argsort(pred_scores)[-k:]
#     top_k_true_labels = true_labels[top_k_indices]
#     return np.sum(top_k_true_labels) / np.sum(true_labels)

def mrr(labels, predictions):
    if len(labels) != len(predictions):
        raise ValueError("Length of labels and predictions must be equal")

    # Combine labels and predictions, then sort by prediction score in descending order
    combined = list(zip(labels, predictions))
    combined_sorted = sorted(combined, key=lambda x: x[1], reverse=True)

    # Identify the rank position of the first relevant item (label == 1)
    for idx, (label, _) in enumerate(combined_sorted):
        if label == 1:
            return 1.0 / (idx + 1)

    # If no relevant item is found, return 0
    return 0.0
# Assuming article_embeddings's second dimension size is 250
input_dim = X_train.shape[1]
content_based_model = create_content_based_model(input_dim)

content_based_model.summary()
X_train
class CustomMetricsCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"\n\nEpoch {epoch+1}:", end=" ")
        for key, value in logs.items():
            print(f"\n- {key}: {value:.4f}", end=", ")
        print("\n")

# Using the custom callback
custom_metrics_callback = CustomMetricsCallback()
# Train the model
history = content_based_model.fit(
    X_train, 
    y_train, 
    epochs=10, 
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[custom_metrics_callback]
)
def evaluate_model_optimized(model, user_profiles_df_train, articles_df, articles_embeddings_df, k=10, num_users=2000):
    embeddings_dict = articles_embeddings_df.T.to_dict('list')
    precisions = []
    recalls = []
    mrrs = []
    ndcgs = []
    
    all_true_labels = []
    all_scores = []

    sampled_user_profiles_df = user_profiles_df_train.sample(n=num_users, random_state=42)
    
    for _, user in tqdm(sampled_user_profiles_df.iterrows(), total=num_users, desc="Evaluating", ncols=100):
        user_embedding = user['user_embedding']
        user_id = user['user_id']
        clicked_articles = set(user['click_article_id'])

        all_embeddings = []
        article_ids = []
        for article_id in articles_df['article_id']:
            if article_id in embeddings_dict:
                article_embedding = embeddings_dict[article_id]
                combined_features = np.concatenate((user_embedding, article_embedding)).reshape(1, -1)
                all_embeddings.append(combined_features)
                article_ids.append(article_id)
        
        all_embeddings = np.vstack(all_embeddings)
        scores = model.predict(all_embeddings, verbose=0).flatten()  # Set verbose=0 to suppress model output
        true_labels = np.array([1 if article_id in clicked_articles else 0 for article_id in article_ids])

        precisions.append(precision_at_k(true_labels, scores, k))
        recalls.append(recall_at_k(true_labels, scores, k))
        mrrs.append(mrr(true_labels, scores))
        ndcgs.append(ndcg_at_k(true_labels, scores, k))
        
        all_true_labels.extend(true_labels)
        all_scores.extend(scores)

    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_mrr = np.mean(mrrs)
    avg_ndcg = np.mean(ndcgs)
    g_auc = roc_auc_score(all_true_labels, all_scores)

    return avg_ndcg, avg_mrr, avg_precision, avg_recall, g_auc
# X_test, y_test = prepare_training_data(user_test_profiles_df, df_articles, article_embeddings_df)
# ndcg_score, mrr_score, auc_score, y_pred = evaluate_model_on_test_data(content_based_model, X_test, y_test)
# print(f"NDCG@10: {ndcg_score:.4f}, MRR: {mrr_score:.4f}, AUC: {auc_score:.4f}")
# Evaluation
# ndcg_score, mrr_score, g_auc_score, y_true, y_pred, user_ids = evaluate_model_optimized(content_based_model, user_profiles_df_test, df_articles, article_embeddings_df, k=10, num_users=len(user_profiles_df_test))
avg_ndcg, avg_mrr, avg_precision, avg_recall, g_auc = evaluate_model_optimized(content_based_model, user_profiles_df_test, df_articles, article_embeddings_df, k=10, num_users=10)
print(f"NDCG@10: {avg_ndcg:.4f}, MRR: {avg_mrr:.4f}, precision: {avg_precision:.4f}, recall: {avg_recall:.4f}, g_auc: {g_auc:.4f}")
# Evaluating: 100%|███████████████████████████████████████████████████| 10/10 [00:08<00:00,  1.13it/s]
# NDCG@10: 0.3124, MRR: 0.3869, precision: 0.1400, recall: 0.4244, g_auc: 0.9683
def infer_top_k_articles(user_id, user_profiles_df, df_articles, article_embeddings_df, model, k=5):
    tmp_df_articles = df_articles.copy()
    # Retrieve the user's embedding
    user_profile = user_profiles_df[user_profiles_df['user_id'] == user_id].iloc[0]
    
    if user_profile.empty:
        raise ValueError("User ID not found in the user profiles.")

    user_embedding = user_profile['user_embedding']

    # Get all articles embeddings
    embeddings_dict = article_embeddings_df.T.to_dict('list')
    
    article_ids = []
    combined_features_list = []
    
    for article_id, article_embedding in embeddings_dict.items():
        article_ids.append(article_id)
        combined_features = np.concatenate((user_embedding, article_embedding)).reshape(1, -1)
        combined_features_list.append(combined_features)

    all_embeddings = np.vstack(combined_features_list)
    
    # Predict relevance scores using the trained model
    scores = model.predict(all_embeddings, verbose=0).flatten()

    print(user_profile["click_article_id"])
    # Add scores to dataframe
    tmp_df_articles['score'] = tmp_df_articles['article_id'].map(dict(zip(article_ids, scores)))
    tmp_df_articles = tmp_df_articles.sort_values(by='score', ascending=False)

    top_articles = tmp_df_articles.copy()[["article_id","category_id","score"]]
    user_article_clicked = top_articles[top_articles['article_id'].isin(user_profile["click_article_id"])].reset_index(drop=True)

    top_articles = top_articles[~top_articles['article_id'].isin(user_profile["click_article_id"])]

    # Rank articles based on scores
    top_k_indices = np.argsort(scores)[-k:][::-1]
    top_k_article_ids = [article_ids[i] for i in top_k_indices]
    
    # Rank articles based on scores (worst)
    bottom_k_indices = np.argsort(scores)[:k]
    bottom_k_article_ids = [article_ids[i] for i in bottom_k_indices]

    # Fetch top K articles metadata
    top_k_articles = top_articles[top_articles['article_id'].isin(top_k_article_ids)].reset_index(drop=True)
    bottom_k_article_ids = top_articles[top_articles['article_id'].isin(bottom_k_article_ids)].reset_index(drop=True)
    bottom_k_article_ids = bottom_k_article_ids.sort_values(by='score', ascending=True)
    
    # Display the top K articles usi
    return top_k_articles, bottom_k_article_ids, user_article_clicked
user_profiles_df_train
user_id=3
user_id=4
top_k_articles, bottom_k_article_ids, user_article_clicked = infer_top_k_articles(user_id, user_profiles_df_all, df_articles, article_embeddings_df, content_based_model, k=5)
user_article_clicked
top_k_articles
bottom_k_article_ids
content_based_model.save('./input/content-based-reduced.h5')
# Save DataFrames to disk
user_profiles_df_all.to_pickle("./input/user_profiles_df_all-reduced.pkl")
df_articles.to_pickle("./input/df_articles-reduced.pkl")
article_embeddings_df.to_pickle("./input/article_embeddings_df-reduced.pkl")
article_embeddings_df.shape
def infer_all_articles_scores(user_id, df, df_articles, article_embeddings_df, model):
    # Retrieve the user's embedding
    user_profile = df[df['user_id'] == user_id].iloc[0]
    
    if user_profile.empty:
        raise ValueError("User ID not found in the user profiles.")

    user_embedding = user_profile['user_embedding']

    # Get all articles embeddings
    embeddings_dict = article_embeddings_df.T.to_dict('list')
    
    article_ids = list(embeddings_dict.keys())
    combined_features_list = [np.concatenate((user_embedding, article_embedding)).reshape(1, -1) 
                              for article_embedding in embeddings_dict.values()]

    all_embeddings = np.vstack(combined_features_list)
    
    # Predict relevance scores using the trained model
    scores = model.predict(all_embeddings, verbose=0).flatten()

    # Create a dataframe with article IDs, category IDs, and scores
    article_scores_df = df_articles[['article_id', 'category_id']].copy()
    article_scores_df['score'] = article_scores_df['article_id'].map(dict(zip(article_ids, scores)))
    
    # Remove any unwanted header rows if present
    # article_scores_df.columns = article_scores_df.columns.droplevel(0)
    article_scores_df.reset_index(drop=True, inplace=True)
    return article_scores_df
articles_scores = infer_all_articles_scores(user_id, user_profiles_df_all, df_articles, article_embeddings_df, content_based_model)
articles_scores

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