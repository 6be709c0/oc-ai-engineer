import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Embedding, Dot, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from scipy.sparse import csr_matrix

# Step 3: Model 1 - Collaborative Filtering using LightGCN
class LightGCN(Model):
    def __init__(self, num_users, num_items, embedding_size=64, n_layers=3, n_components=16, reg=1e-5):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size
        self.n_layers = n_layers
        self.n_components = n_components
        self.reg = reg

        # User and item embedding layers
        self.user_embedding = Embedding(num_users, embedding_size, name='user_embedding')
        self.item_embedding = Embedding(num_items, embedding_size, name='item_embedding')

        # LightGCN layers
        self.gcn_layers = self.build_gcn_layers()

    def build_gcn_layers(self):
        gcn_layers = []
        for layer_id in range(self.n_layers):
            gcn_layers.append(self.gcn_layer(layer_id))
        return gcn_layers

    def gcn_layer(self, layer_id):
        return [
            Embedding(self.num_users, self.n_components, name=f'gcn_user_embed_{layer_id}', embeddings_regularizer=self.reg),
            Embedding(self.num_items, self.n_components, name=f'gcn_item_embed_{layer_id}', embeddings_regularizer=self.reg),
            Dot(axes=2, name=f'gcn_dot_{layer_id}'),
            Flatten(name=f'gcn_flatten_{layer_id}'),
            Dense(1, activation='linear', name=f'gcn_dense_{layer_id}')
        ]

    def call(self, inputs):
        user_id, item_id = inputs

        # Embedding
        user_embed = self.user_embedding(user_id)
        item_embed = self.item_embedding(item_id)

        # LightGCN layers
        output = user_embed * item_embed
        for user_embed_gcn, item_embed_gcn, dot, flatten, dense in self.gcn_layers:
            user_embed_gcn = user_embed_gcn(user_id)
            item_embed_gcn = item_embed_gcn(item_id)
            output = output + dot([user_embed_gcn, item_embed_gcn])
            output = flatten(output)
            output = dense(output)

        return output