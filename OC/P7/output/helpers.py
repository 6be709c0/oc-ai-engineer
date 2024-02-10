# Standard library imports  
import os  
import json  
import re  
import string  
  
# Data handling and visualization  
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
  
# Natural Language Processing and text processing  
import nltk  
import demoji  
from sklearn.feature_extraction.text import CountVectorizer  
from gensim.models import Word2Vec  
from keras.preprocessing.text import Tokenizer, tokenizer_from_json  
  
# Machine Learning - Preprocessing and evaluation  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score  
from sklearn.linear_model import LogisticRegression  
from tensorflow.keras.preprocessing.sequence import pad_sequences  
  
# Deep Learning framework and layers  
import tensorflow as tf  
from tensorflow.keras import Sequential  
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D, Dense, LSTM, Conv1D, Embedding  
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping  
  
# Model persistence and parallel processing  
import pickle  
from joblib import Parallel, delayed  
  
# Experiment tracking and Hyperparameter optimization  
import mlflow  
import mlflow.keras
from mlflow.models import infer_signature  
from hyperopt import hp, STATUS_OK, fmin, tpe, Trials  
  
# PyTorch data handling
from torch.utils.data import Dataset  
  
# Utility and progress bar  
from tqdm import tqdm

def clean_df(dataframe):
    df = dataframe.copy()
    # Keep only comment and sentiment columns
    df = df[["comment","sentiment"]]
    
    # negative field 0 = 0
    # Map positive field 4 = 1
    df.loc[df['sentiment'] == 4, 'sentiment'] = 1  
    
    # Clean the comment
    df['comment_clean'] = parallelize_on_rows(df['comment'], clean_tweet)  
    
    # Count the number of words from comment & comment_cleam
    df['words_nb'] = parallelize_on_rows(df['comment'], lambda x: len(x.split()))  
    df['words_nb_clean'] = parallelize_on_rows(df['comment_clean'], lambda x: len(x.split()))  
    
    # Only keep the clean words
    df = df[df['words_nb_clean'] > 3]
    
    # Remove duplicate
    df.drop_duplicates(subset='comment',inplace=True)
    
    return df    
    
def clean_tweet(doc):  
    # Lower the code
    doc = doc.lower().strip()
    #remove emoji
    text = demoji.replace(doc, '')
    #remove links
    text = re.sub(r'http\S+|www.\S+', '', text)  
    # # Remove mentions
    text = re.sub(r'@\w+', '', text) 
    # Remove hashtag symbol but keep the text  
    text = re.sub(r'#(\w+)', r'\1', text)
    # Keep only alphanumeric characters and spaces  
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove multiple spaces (replace them with a single space)  
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
    
def parallelize_on_rows(data, func):  
    r = Parallel(n_jobs=-1)(delayed(func)(i) for i in tqdm(data, desc="Processing"))  
    return r    

def get_data(config, dataframe, get_embedding_model, get_embedding_matrix):
    df = dataframe.copy()
    
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(df["comment_clean"], df['sentiment'], test_size=0.3, random_state=42)
    
    # Prepare the corpus of document for the tokenizer
    X_data = np.array(df['comment_clean'])
    
     # Prepare data for Word2Vec (split sentences into words)  
    X_train_tokens = [text.split() for text in X_train]  
    
    # Prepare embedding model
    embedding_model = get_embedding_model(X_train_tokens, config)

    # Tokenizer setup
    tokenizer = Tokenizer(filters="", lower=False, oov_token="<oov>")
    tokenizer.fit_on_texts(X_data)
    tokenizer.num_words = config["vocab_length"]
    
    # Padding the text
    X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=config["input_length"])
    
    # Creating the embedding matrix
    embedding_matrix = get_embedding_matrix(embedding_model, config, tokenizer)
    
    # Getting the model
    model = getModel(config, embedding_matrix)
    
    # Setting the callback
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', patience=2, cooldown=0),
        EarlyStopping(monitor='val_accuracy', min_delta=1e-4, patience=2),
    ]
    
    return (model, tokenizer, callbacks, X_train_pad, y_train)

def get_w2vec_model(X_train_tokens, config):
    model = Word2Vec(sentences=X_train_tokens, vector_size=config["vector_size"], window=config["window"], workers=config["workers"])  
    print("W2vec Vocabulary Length (you can adjust the vocal length):", len(model.wv.key_to_index))
    return model

def get_glove_model(X_train_tokens, config):
    # Make sure to download the glove model beforehand
    # https://nlp.stanford.edu/projects/glove/
    glove_path="glove/glove.twitter.27B/glove.twitter.27B.100d.txt"
    embeddings_index = {}
    with open(f'{glove_path}', encoding='utf8') as f:  
        for line in f:  
            values = line.split()  
            word = values[0]  
            coefs = np.asarray(values[1:], dtype='float32')  
            embeddings_index[word] = coefs
    print("Glove Vocabulary Length (you can adjust the vocal length):", len(embeddings_index))
    return embeddings_index

def get_w2vec_embedding_matrix(model, config, tokenizer):
    # Creating the embedding matrix
    embedding_matrix = np.zeros((config["vocab_length"], config["vector_size"]))
    for word, token in tokenizer.word_index.items():
        if model.wv.__contains__(word):
            embedding_matrix[token] = model.wv.__getitem__(word)
    print("Embedding Matrix Shape:", embedding_matrix.shape)
    return embedding_matrix

def get_glove_embedding_matrix(model, config, tokenizer):
    # Creating the embedding matrix
     # Creating the embedding matrix
    embedding_matrix = np.zeros((config["vocab_length"], config["vector_size"]))  
    for word, i in tokenizer.word_index.items():  
        if i <  config["vocab_length"]:  
            embedding_vector = model.get(word)  
            if embedding_vector is not None:  
                embedding_matrix[i] = embedding_vector  
    print("Embedding Matrix Shape:", embedding_matrix.shape)
    return embedding_matrix


def get_w2vec_data(config, dataframe):
    return get_data(
        config, 
        dataframe, 
        get_w2vec_model, 
        get_w2vec_embedding_matrix
    )
    

def get_glove_data(config, dataframe):
    return get_data(
        config, 
        dataframe, 
        get_glove_model, 
        get_glove_embedding_matrix
    )
    
    
def getModel(config, embedding_matrix):
    embedding_layer = Embedding(input_dim=config["vocab_length"],
                                output_dim=config["vector_size"],
                                weights=[embedding_matrix],
                                input_length=config["input_length"],
                                trainable=False)

    model = Sequential([
        embedding_layer,
        Bidirectional(LSTM(config["vector_size"], dropout=0.3, return_sequences=True)),
        Conv1D(config["vector_size"], 5, activation='relu'),
        GlobalMaxPool1D(),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid'),
    ],
    name="Sentiment_Model")
    return model

  
def objective(params, X, y, model, callbacks, tokenizer, log=True):  
      
    print("Running one fit with the params: ", params)
    mlflow.set_experiment(params["experiment_name"])  
          
    with mlflow.start_run():  
        
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=params['learning_rate'])  
          
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
           
        # Tensorflow prediction discrepancy with GPU VS CPU.
        # Fixed in tf-nightly, waiting for a fix fully deployed in new tensorflow version
        with tf.device('/CPU:0'): 
            history = model.fit(    
                X, y,    
                batch_size=int(params['batch_size']),    
                epochs=int(params['epochs']),    
                validation_split=0.1,   
                callbacks=callbacks,    
                verbose=1,    
            )
          
            pred = model.predict(X)  
            pred = np.where(pred >= 0.5, 1, 0)  
            
            signature = infer_signature(X, pred)  
            
            if(log):
                mlflow.log_params(params)    
                mlflow.log_metric("best_val_accuracy", max(history.history['val_accuracy']))    
          
        # with open('tokenizer.json', 'w', encoding='utf-8') as f:    
        #     f.write(json.dumps(tokenizer.to_json(), ensure_ascii=False))    
          
        with open("./tokenizer.pkl", "wb") as f:  
            pickle.dump(tokenizer, f)
          
        with open("./model.pkl", "wb") as f:  
            pickle.dump(model, f)
            
        with open('params.json', 'w', encoding='utf-8') as f:    
            f.write(json.dumps({
                                "input_length": int(params["input_length"])
                                }, 
                               ensure_ascii=False))    
        
        if(log):
            # mlflow.keras.log_model(model, "model", signature=signature)  
            mlflow.log_artifact('tokenizer.pkl')
            mlflow.log_artifact('model.pkl')
            mlflow.log_artifact('params.json')
            print("Log done")  
          
        return {'loss': -max(history.history['val_accuracy']), 'status': STATUS_OK}  
  