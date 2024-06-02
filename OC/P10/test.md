Explique moi SVD simplement puis à partir de ce code
```python
import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict

from helpers import load_dataset
# Load datasets
df_articles, df_clicks, _ = load_dataset()
# Create user profiles based on article clicks
user_profiles = df_clicks.groupby('user_id')['click_article_id'].apply(list).reset_index()
article_category_map = df_articles.set_index("article_id")["category_id"].to_dict()
user_profiles["categories"] = user_profiles["click_article_id"].apply(
    lambda x: [article_category_map[article_id] for article_id in x]
)
user_profiles
# Merge datasets to get user-article-category information
df_merged = df_clicks.merge(df_articles, left_on='click_article_id', right_on='article_id')
df_user_item = df_merged[['user_id', 'article_id', 'category_id']]
# Create user-article-category interaction counts
interaction_counts = df_user_item.groupby(['user_id', 'article_id']).size()
# Convert series to dataframe and reset index
user_rating_matrix = interaction_counts.to_frame().reset_index()
user_rating_matrix.rename(columns={0: 'rating'}, inplace=True)
##### Normalize ratings #####
scaler = MinMaxScaler(feature_range=(0, 1))
user_rating_matrix["rating_norm"] = scaler.fit_transform(
    np.array(user_rating_matrix["rating"]).reshape(-1, 1)
)
# Filter out zero normalized ratings
X = user_rating_matrix[user_rating_matrix["rating_norm"] != 0.0]
X

##### Prepare dataset for Surprise library #####
reader = Reader(rating_scale=(0, 1))
data = Dataset.load_from_df(X[["user_id", "article_id", "rating_norm"]], reader)
trainset, testset = train_test_split(data, test_size=0.25)
print("Number of interactions: ", len(X))
# Train the SVD model and evaluate #####
svd = SVD()
svd.fit(trainset)
predictions = svd.test(testset)
# Perform cross-validation with additional metrics
def compute_metrics(predictions):
    """Compute various evaluation metrics from the predictions."""
    # Calculate RMSE and MAE
    metrics = {
        'rmse': accuracy.rmse(predictions, verbose=False),
        'mae': accuracy.mae(predictions, verbose=False),
        'ndcg_5': ndcg_at_k(predictions, k=5),
        'ndcg_10': ndcg_at_k(predictions, k=10),
        'mean_mrr': mean_reciprocal_rank(predictions)
    }
    return metrics
def get_top_n(predictions, n=10):
    """Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions (list of Prediction objects): The list of predictions, as
        returned by the test method of an algorithm.
        n (int): The number of recommendation to output for each user. Default is 10.

    Returns:
        dict: A dictionary where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, category id, rating estimation), ...] of size n.
    """
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, article_category_map[iid], est))

    # Then sort the predictions for each user and retrieve the n highest ones
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[2], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n
top_recommendations = get_top_n(predictions)
top_recommendations
def ndcg_at_k(predictions, k=5, relevance_threshold=0.1):
    from collections import defaultdict
    import math
    
    user_est_true = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))
    
    ndcg = 0.0
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        
        print(f"\nUser: {uid}")
        print(f"Sorted ratings (estimated, true): {user_ratings}")
        
        dcg = 0.0
        idcg = 0.0
        
        relevant_ratings = [true_r for est, true_r in user_ratings if true_r > relevance_threshold]
        for i in range(min(len(relevant_ratings), k)):
            idcg += 1.0 / math.log2(i + 2)
        
        for i, (est, true_r) in enumerate(user_ratings[:k]):
            if true_r > relevance_threshold:
                dcg += 1.0 / math.log2(i + 2)
            print(f"Rank: {i+1}, Estimated Rating: {est}, True Rating: {true_r}, DCG: {dcg}")

        ndcg_value = dcg / idcg if idcg > 0 else 0
        ndcg += ndcg_value
        print(f"User NDCG: {ndcg_value}, IDCG: {idcg}")

    final_ndcg = ndcg / len(user_est_true)
    print(f"\nAverage NDCG@{k}: {final_ndcg}")
    return final_ndcg

def mean_reciprocal_rank(predictions, relevance_threshold=0.1):
    from collections import defaultdict
    
    user_est_true = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))
    
    mrr = 0.0
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        
        print(f"\nUser: {uid}")
        print(f"Sorted ratings (estimated, true): {user_ratings}")
        
        for rank, (est, true_r) in enumerate(user_ratings):
            if true_r > relevance_threshold:
                mrr += 1.0 / (rank + 1)
                print(f"First relevant rank: {rank+1}, MRR contribution: {1.0/(rank+1)}")
                break

    final_mrr = mrr / len(user_est_true)
    print(f"\nMean Reciprocal Rank: {final_mrr}")
    return final_mrr
compute_metrics(predictions)
predictions_all = svd.test(data.build_full_trainset().build_testset())
all_recommendations = get_top_n(predictions_all)
def sort_users_by_highest_score(user_scores):
    user_max_scores = []

    for user, scores in user_scores.items():
        if scores:  # Check if the list is non-empty
            highest_score = max(scores, key=lambda x: x[2])[2]
            user_max_scores.append((user, highest_score))

    sorted_users = sorted(user_max_scores, key=lambda x: x[1], reverse=True)
    return sorted_users
sorted_users_by_highest_score = sort_users_by_highest_score(all_recommendations)
```