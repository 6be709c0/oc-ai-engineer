from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import traceback
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA


with open('input/archive/articles_embeddings.pickle', 'rb') as file:
    article_embeddings = pickle.load(file)

k = 5  # number of top similarities to keep
top_k_indices = np.argsort(similarity_matrix, axis=1)[:, -k:]
top_k_values = np.array([
    similarity_matrix[i, top_k_indices[i]] for i in range(similarity_matrix.shape[0])
])

# Alternative compact way to look up values is by using advanced indexing:
# top_k_values = similarity_matrix[np.arange(similarity_matrix.shape[0])[:, None], top_k_indices]