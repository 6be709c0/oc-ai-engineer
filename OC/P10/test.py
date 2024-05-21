from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import traceback
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA

# # Load cosine_similarities back from a file
# loaded_cosine_similarities = np.load('cosine_similarities.npy')
# print(loaded_cosine_similarities)

with open('reduced.pkl') as file:
# with open('input/archive/articles_embeddings.pickle', 'rb') as file:
    article_embeddings = pickle.load(file)

print("AA", article_embeddings.shape)
article_embeddings = article_embeddings[:10000]

pca = PCA(n_components=0.2)
reduced_embeddings = pca.fit_transform(article_embeddings)
print(reduced_embeddings.shape)

with open('reduced.pkl', 'wb') as f:
    pickle.dump(reduced_embeddings, f)
# reduced_embeddings = article_embeddings

# print("Test")
# cosine_similarities  = cosine_similarity(reduced_embeddings)
# cosine_similarities = cosine_similarities.astype(np.float32)

# print("Test Done")
# # # Save cosine_similarities to a file
# np.save('cosine_similarities.npy', cosine_similarities)
