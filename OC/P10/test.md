```python
##### Imports and Dataset Loading #####
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import pickle
import json
from helpers import *
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
from keras.metrics import RootMeanSquaredError

# Load datasets
df_articles, df_clicks, article_embeddings = load_dataset()
##### PCA and Variance Plotting #####
# Fit PCA
pca = PCA()
pca.fit(article_embeddings)

# Variance data
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
components = np.arange(len(cumulative_variance)) + 1

# Plotting
plt.figure(figsize=(10, 4))
plt.plot(components, cumulative_variance, label='Cumulative Explained Variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance (%)')
plt.title('PCA Explained Variance')

# Annotate specific variance percentages
variance_thresholds = [0.9, 0.95, 0.97, 0.98, 0.99]
for threshold in variance_thresholds:
    component_number = np.where(cumulative_variance >= threshold)[0][0]
    plt.scatter(component_number + 1, cumulative_variance[component_number], color='red')
    plt.annotate(f"{int(threshold * 100)}%", 
                 (component_number + 1, cumulative_variance[component_number]),
                 textcoords="offset points", 
                 xytext=(0, 10), ha='center')

plt.grid(True)
plt.show()

# PCA with 98% variance
pca = PCA(n_components=0.98)
reduced_embeddings = pca.fit_transform(article_embeddings)
article_embeddings = reduced_embeddings
##### Preprocessing #####
# Preprocess data
df_articles = preprocessing_articles(df_articles)
df_clicks = preprocessing_clicks(df_clicks)
articles_embed_df = pd.DataFrame(article_embeddings)

# Filter articles clicked
articles_clicked = df_clicks.click_article_id.value_counts().index
df_articles = df_articles.loc[articles_clicked]
articles_embed_df = articles_embed_df.loc[articles_clicked]

# Print shapes
print("df_articles shape:", df_articles.shape)
print("article_embeddings shape:", articles_embed_df.shape)
df_articles = df_articles.sample(n=5000)
df_clicks = df_clicks.sample(n=50000)
##### Session-based Train-Test Split #####
def train_test_split_sessions(clicks_df, test_size=0.1, val_size=0.1, random_state=42):
    session_ids = clicks_df['session_id'].unique()
    train_sessions, test_sessions = train_test_split(session_ids, test_size=test_size, random_state=random_state)
    train_sessions, val_sessions = train_test_split(train_sessions, test_size=val_size, random_state=random_state)
    train_df = clicks_df[clicks_df['session_id'].isin(train_sessions)]
    val_df = clicks_df[clicks_df['session_id'].isin(val_sessions)]
    test_df = clicks_df[clicks_df['session_id'].isin(test_sessions)]
    return train_df, val_df, test_df

# Reduced sample for quicker model building
# df_clicks = df_clicks.sample(n=500000)
train_clicks_df, val_clicks_df, test_clicks_df = train_test_split_sessions(df_clicks)

print(f"Training clicks shape: {train_clicks_df.shape}")
print(f"Validation clicks shape: {val_clicks_df.shape}")
print(f"Testing clicks shape: {test_clicks_df.shape}")
##### User Profile Creation #####
def create_user_profiles(clicks_df, article_embeddings_df):
    user_profiles = clicks_df.groupby('user_id')['click_article_id'].apply(list).reset_index()
    embeddings_dict = article_embeddings_df.T.to_dict('list')
    
    user_profiles['user_embedding'] = user_profiles['click_article_id'].progress_apply(
        lambda x: np.mean([embeddings_dict[article] for article in x if article in embeddings_dict], axis=0)
    )
    
    return user_profiles

tqdm.pandas()
user_profiles_train = create_user_profiles(train_clicks_df, articles_embed_df)
user_profiles_val = create_user_profiles(val_clicks_df, articles_embed_df)
user_profiles_test = create_user_profiles(test_clicks_df, articles_embed_df)
user_profiles_all = create_user_profiles(df_clicks, articles_embed_df)
def create_content_based_model(input_dim):
    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

##### Data Preparation #####
def prepare_data(user_profiles_df, articles_df, articles_embed_df):
    X, y = [], []
    embeddings_dict = articles_embed_df.T.to_dict('list')

    for _, user in tqdm(user_profiles_df.iterrows(), total=len(user_profiles_df)):
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
                
    return np.array(X), np.array(y)

pickle_file = 'output/Xy_train_val.pkl'
if os.path.exists(pickle_file):
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
        X_train, y_train, X_val, y_val = data['X_train'], data['y_train'], data['X_val'], data['y_val']
else:
    X_train, y_train = prepare_data(user_profiles_train, df_articles, articles_embed_df)
    X_val, y_val = prepare_data(user_profiles_val, df_articles, articles_embed_df)
    
    with open(pickle_file, 'wb') as f:
        pickle.dump({'X_train': X_train, 'y_train': y_train, 'X_val': X_val, 'y_val': y_val}, f)
##### Model Training #####
input_dim = X_train.shape[1]
content_based_model = create_content_based_model(input_dim)

content_based_model.compile(
    optimizer=tf.keras.optimizers.legacy.Adam(),
    loss='binary_crossentropy',
    metrics=[RootMeanSquaredError()])

# Train the model
history = content_based_model.fit(
    X_train, 
    y_train, 
    epochs=10, 
    batch_size=32,
    validation_data=(X_val, y_val),
)

##### Evaluate model #####
```