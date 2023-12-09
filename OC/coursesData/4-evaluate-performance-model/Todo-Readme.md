Goal: 

- TP on ECG with update ridge
- Update that on the graph
- TP on perceptrons
- Add an sample into the exercise
- TP on bagging
- Add an sample into the exercise
- TP on randomforest
- Add an sample into the exercise

----
CHATGPT Queues:

In this code, how to add the execution time of the fit function in the last print:
```
print(f"Entrainement du modèle...")
rf_model.fit(X_train, y_train)
print(f"Modèle entrainé en 00:00:00")
```
---
In this code: 
```

# Create a 1x2 grid of subplots  
fig, axs = plt.subplots(1, 2, figsize=(12, 3))  
  
# Scatter plot  
sns.scatterplot(data=prds, x='index', y='DAYS_EMPLOYED', ax=axs[0])  
axs[0].set_title('Scatter Plot de DAYS_EMPLOYED')  
axs[0].set_xlabel('Index')  
axs[0].set_ylabel('DAYS_EMPLOYED')  
  
# Histogram  
app_train['DAYS_EMPLOYED'].plot.hist(ax=axs[1], bins=50)  
axs[1].set_title('Histogram de DAYS_EMPLOYED')  
axs[1].set_xlabel('DAYS_EMPLOYED')  
axs[1].set_ylabel('Fréquence')  
  
# Adjust the spacing between subplots  
plt.tight_layout()  
  
# Show the plot  
plt.show()  
```

How to have the first graph taking 8 of the width and the second graph 4 ?

---

What does oob_score mean in `rfc2 = RandomForestClassifier(n_estimators=500, oob_score=True)`

---
1. Checkbox all the courses
   1. Make sure to check all videos of all the courses.
2. Do all TP.

3. Understand polynoimial features > 
    > https://jakevdp.github.io/PythonDataScienceHandbook/05.04-feature-engineering.html

4. Lasso Ridge Improved > https://jakevdp.github.io/PythonDataScienceHandbook/05.06-linear-regression.html

---

Update Notebook > 
1. Version with:
- Keep as is
- Set to 0 to check the weight
- Set to -1 and move all to positive
```
app_train['DAYS_EMPLOYED'].replace({365243: np.NaN}, inplace = True) # TODO (You can normalize those data, is has to be positive)
```

2. Update globally just for the graph
```
app_train.drop(app_train[app_train["AMT_INCOME_TOTAL"] >= 500000.0].index, inplace=True) # TODO just for the graph
```

3. You should ask GPT for why should I need to do that.
Enough to do with point 1.
```python
app_train['DAYS_ID_PUBLISH'] = abs(app_train['DAYS_ID_PUBLISH']) # Todo check bias
```

4. Keeping records
Enoguh with point 1
```
# Todo 
# Every data that you want to keep, modify it to positive
# Every data thagt you don't to delete, modify to -1
# Todo 2 > Instead of -1, put to 0 to check the weight
```

5. Update globally. Add a table of score next to feature importance
Not sure I understand what I wrote by "each columns"
```
# TODO Table with score next to each columns (AGE & Gender)
```

6. Correlation should be 1 max. Ask GPT
```
# Why 1.12, shouldn't the sum be 1 ?
# Todo, understand this
```

7. Remove TARGET
```
# Todo > Remove target accordingly
test_corr.drop(columns=["TARGET"], inplace=True)
Doesn't work
```

8. Graph
```
# Todo > Cumulative
```

9. Global
```
# TODO >
# Features importances global
# Entraine un random forest avec toutes les cases, c'est mieux de les normaliser
# Excplication métier > Gain de temps en rendez-vous car moins de features.
```


10. PCA Improved > https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html

Later > Model Validation > https://jakevdp.github.io/PythonDataScienceHandbook/05.03-hyperparameters-and-model-validation.html


What is going to be important as well is to determine the score to predict if the target is 0 or 1 by features! SO to make it 1, you should say that the feature A had a weight of x and feature B had a weight of y, etc.