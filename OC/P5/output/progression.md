# Progression

- ✅ Notebook analyse exploratoire 
- ⏳ Notebook approches de modélisation
- ⏳ Notebook calcul fréquence mise à jour
- Support de présentation

---

## Modélisation

- ✅ Ajouter la satisfaction
- ✅ Enlever une partie des données
  - ✅50%/50%
  - ✅ Quantité égale [2_modelisation_all_undersample](./2_modelisation_all_undersample.ipynb), [2_modelisation_all_undersample_score](./2_modelisation_all_undersample_score.ipynb)
- ✅ Ajouter la catégories de produits
- ✅ Visualisation comme Github
- ✅ Ajout du graph silhouette [silhouette](https://www.kaggle.com/code/kautumn06/yellowbrick-clustering-evaluation-examples/notebook)
- ⏳ Créer un score de modèle pour choisir le mieux
- Add graph to show the difference between (recency vs frequency, recency_score vs avg_satisfaction, ...)

QUESTION:
-  Évalué la stabilité des clusters à l’initialisation ? What does it mean ? That it change if I rerun it ?

TODO TUESDAY:
- Cluster graph like in https://www.kaggle.com/code/hamadizarrouk/segmentation-des-clients-d-un-site-e-commerce-nb2#Segmentation-des-clients-du-site-E-Commerce-Olist
  - One for cluster size
  - One for features?
- Start to think about how to analyse the model stability in time to check the maintenance
  - Inital category
  - Try to merge with explainability from other source?
LATER:
- Add multiple model and other hyperparameter
- Evaluate best model depending of values (score, training time, see if the avg silhouette score is better too, ...)
- Add graph explaining best model usage during the notebook_2
- Maybe I have outliers too

- At the end, I think the best use case it to define cluster without categories, then adding the most important categories within each cluster. Like a pie chart

- I need one file(one dataset) where I will test different hyperparameters for:
kmeans, dbscan, tsne, with and without pca, print the score of all in a graph.
And choose the best one, define the clusters definition within each.
Add the top categories within each.
Calculate the simulation time by using the difference of cluster over time per customer as score.

## To show

- `1_analyse_exploratoire`
  - Correct review_score
    - Some were duplicated (got the latest one)
    - I actually have 768 orders without reviews (set as -1)
    - Added delivery mean delay
    - Added RFM score as R+F+M as 1+2+3=123
    - Added categories
    - Explain the graph you wanna implement to compare recency vs frequency, ...
- Only RFM `2_modelisation_all`
- Only RFM undersample `2_modelisation_all_undersample` > **Silhouette**

---

- Only RFM with scoring (1-5) `2_modelisation_all_score`
- Only RFM with scoring (1-5) undersample `2_modelisation_all_undersample_score`

---


- Adding delivery `2_modelisation_delivery`
- Adding delivery undersample `2_modelisation_delivery_undersample` 0.5
- Adding delivery norm undersample `2_modelisation_delivery_undersample_norm`
- Adding delivery score all undersample 
  `2_modelisation_delivery_undersample_weird_norm`

---

- Adding delivery undersample PCA `2_modelisation_delivery_undersample_pca` 0.5
- Adding delivery norm undersample PCA `2_modelisation_delivery_undersample_norm_pca`
- Adding delivery score all undersample PCA `2_modelisation_delivery_undersample_weird_norm_pca`

---

- Adding categories undersample norm pca `2_modelisation_full_all_undersample` > 0.5 !
- Adding categories norm pca `2_modelisation_full_all` ~0.32
- Adding categories norm pca `2_modelisation_full_all_half` ~0.33

---

- graph is like is 2_modelisation_full_all_undersample and 2_modelisation_delivery_undersample

- Graph > sum, max sur les categories, (eps=X, Min Sample = 100)

- Ask ChatGPT to subcategorized in 10 categories


----

TODO Next:

- Add new score based on:
  - data not normalised
  - data with rank on all columns (avg delay, rfm score, categ)
  - data with rank on all columns (avg delay, rfm score, WITHOUT categ)
  - data with different PCA components


- Check whatsapp for better understanding
- What is ARI
- Calculate ARI