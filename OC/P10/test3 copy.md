From this code:
```
from helpers_2 import *
# Load datasets
df_articles, df_clicks = load_dataset()
dataframe = df_clicks.merge(df_articles, left_on='click_article_id', right_on='article_id')
dataframe = dataframe[['user_id', 'article_id', 'category_id']]
dataframe
series = dataframe.groupby(['user_id', 'category_id']).size()
user_rating_matrix = series.to_frame()
user_rating_matrix = user_rating_matrix.reset_index()
user_rating_matrix.rename(columns = {0:'rate'}, inplace = True)
user_rating_matrix["rate"].value_counts()
reader = Reader(rating_scale=(1,10))
_x = user_rating_matrix.loc[user_rating_matrix.rate > 1]
data = Dataset.load_from_df(_x[['user_id', 'category_id', 'rate']], reader)

print('We have selects', len(_x), 'interactions.')
trainset, testset = train_test_split(data, test_size=0.25)
print('Test set lenght :', len(testset))
print('Train set lenght :', len(_x) - len(testset))
from surprise import SVD, accuracy
algo = SVD()
algo.fit(trainset)
predictions = algo.test(testset)
print('Number of predictions in Test set :', len(predictions))
accuracy.rmse(predictions)
```

I want to use the prediction per article to use them with another model.
So can you normalise the prediction score for each article for each user id ?
