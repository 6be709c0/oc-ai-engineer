Optimize that code:
```python
def infer_top_k_articles(user_id, user_profiles_df_train, df_articles, article_embeddings_df, model, k=5):
    tmp_df_articles = df_articles.copy()
    # Retrieve the user's embedding
    user_profile = user_profiles_df_train[user_profiles_df_train['user_id'] == user_id].iloc[0]
    
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
    ```

    And update it to return instead the score of all article per user id