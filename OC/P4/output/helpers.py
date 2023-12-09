from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error  
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer  
from sklearn import svm, metrics

from xgboost import plot_importance as xgb_plot_importance
from xgboost import plot_tree as xgb_plot_tree
from xgboost import XGBClassifier

from imblearn.under_sampling import RandomUnderSampler
from lime.lime_tabular import LimeTabularExplainer
from IPython.display import display  

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import tensorflow as tf
import seaborn as sns
import pandas as pd
import numpy as np

import warnings  
import timeit
import copy
import os

def display_percentage_missing_chart(df, xticksRange):
    # Get the missing data in percentage
    missing_data = df.isna().mean() * 100

    # Sort by value asc
    missing_data = missing_data.sort_values()

    # Create the figure
    plt.figure(figsize=(12, 2))

    ## Add horizontal lines
    sns.set_style('whitegrid')

    ## Set a color palette (https://seaborn.pydata.org/tutorial/color_palettes.html)
    colors = sns.color_palette("rocket_r", len(missing_data))

    ## Create the barplot
    sns.barplot(x=missing_data.index, y=missing_data,  palette=colors) # Create the bar plot

    ## Rotate the x labels and only one out of three for better readability
    plt.xticks(rotation=-90,fontsize=8)
    plt.xticks(range(0, len(missing_data.index), xticksRange))

    ## Set the labels
    plt.ylabel('Pourcentage')
    plt.xlabel('Colonnes')
    plt.title('Pourcentage de valeurs manquantes')
    plt.show()

def create_color_palette(values, thresholds, colors):  
    """Create a color palette based on the given thresholds and colors."""  
    color_palette = []  
    for value in values:  
        for i in range(len(thresholds)):  
            if value < thresholds[i]:  
                color_palette.append(colors[i])  
                break  
        else:  
            color_palette.append(colors[-1])  
    return color_palette

def display_graph_missing_data(df, figsize_height):
    # Calculate the missing value percentage and count for each column  
    missing_data = df.isna().mean() * 100  
    missing_count = df.isna().sum()  
    unique_count = df.nunique()
    # Create a DataFrame to store the missing value percentage and count  
    missing_df = pd.DataFrame({"Missing Percentage": missing_data, "Missing Count": missing_count, "Unique Count": unique_count})  

    # Sort the DataFrame by missing percentage in descending order  
    missing_df = missing_df.sort_values(by="Missing Percentage", ascending=False)  
    thresholds = [25, 50, 75, 100]  
    colors = ["#4fff87", "#4fc4ff", "#ffbc4f", "#ff4f4f"]  

    # Map the colors based on the percentage value  

    color_palette = create_color_palette(missing_df["Missing Percentage"], thresholds, colors)  

    plt.figure(figsize=(10, figsize_height)) # Adjust the figure size as per your preference  
    ax = sns.barplot(x="Missing Percentage", y=missing_df.index, data=missing_df, palette=color_palette) # Create a horizontal bar plot  
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    # Add labels and legend to the plot  
    plt.xlabel("Valeurs manquantes en %")  
    plt.ylabel("Colonnes")  
    plt.title("Pourcentage de valeurs manquantes")  

    # Add the count of missing values inside each bar  
    # for i, (value, name) in enumerate(zip(missing_df["Missing Count"], missing_df.index)):  
    #     ax.text(1, i, f" {value} ", va="center")  
    for i, (missing_value, unique_value, name) in enumerate(  
        zip(missing_df["Missing Count"], missing_df["Unique Count"], missing_df.index)  
    ):
        ax.text(1, i, f"Manquant: {missing_value}", va="center")
        ax.text(20, i, f"Unique: {unique_value}", va="center")
    # Create a custom legend  
    legend_labels = [f"{thresholds[i]-25}-{thresholds[i+1]-25}%" if i != len(thresholds)-1 else f"{thresholds[i]-25}+%" for i in range(len(thresholds))]  
    colors_scaled = [plt.cm.colors.to_rgb(color) for color in colors]  
    legend_elements = [plt.Line2D([0], [0], marker="s", color="white", markerfacecolor=colors_scaled[i], markersize=10) for i in range(len(colors))]  
    plt.legend(legend_elements, legend_labels, loc="lower right")
    
    # Display the plot  
    plt.show()
    
def display_scatter_and_hist_graph_for_column(df, column_name):
    prds = df[column_name].copy().reset_index()  

    # Create a 1x2 grid of subplots  
    fig, axs = plt.subplots(1, 2, figsize=(12, 3))  
        
    # Scatter plot  
    sns.scatterplot(data=prds, x='index', y=column_name, ax=axs[0])  
    axs[0].set_title(f'Scatter Plot de {column_name}')  
    axs[0].set_xlabel('Index')  
    axs[0].set_ylabel(column_name)  
        
    # Histogram  
    df[column_name].plot.hist(ax=axs[1], bins=50)  
    axs[1].set_title(f'Histogram de {column_name}')  
    axs[1].set_xlabel(column_name)  
    axs[1].set_ylabel('Fréquence')  
        
    # Adjust the spacing between subplots  
    plt.tight_layout()  
        
    # Show the plot  
    plt.show()
    
    
def check_corr(df):
    correlations = df.corr(numeric_only=True)
    
    # Filter correlations above 0.8 and remove duplicates  
    high_corr = (correlations.abs() >= 0.9) & (correlations.abs() < 1)  
    high_corr_vars = [(var1, var2) for var1 in high_corr.columns for var2 in high_corr.index if high_corr.loc[var1, var2]] 
    
    # Create a list of dictionaries to store the correlation information  
    correlation_data = []  
    printed_vars = set()  # to keep track of already printed pairs  
      
    # Iterate over the high correlated variables and add the information to the list  
    for var1, var2 in high_corr_vars:  
        # Check if var1 is less than var2 and the pair hasn't been added before  
        if (var1, var2) not in printed_vars and (var2, var1) not in printed_vars:  
            correlation_data.append({  
                'Variable 1': var1,  
                'Variable 2': var2,  
                'Correlation': correlations.loc[var1, var2]  
            })  
            printed_vars.add((var1, var2))  # add the pair to printed_vars  
      
    # Convert the list of dictionaries to a DataFrame  
    correlation_table = pd.DataFrame(correlation_data)  
      
    # Sort the DataFrame by correlation values in descending order  
    correlation_table = correlation_table.sort_values(by='Correlation', ascending=False)  
    correlation_table.reset_index(drop=True, inplace=True)  
    return correlations

def get_target_correlations(df, graph=True):
    dataframe = df.copy()
    correlations = check_corr(dataframe)

    if graph == False:
        return correlations
    
    # Sort the correlations in ascending order  
    sorted_corr = correlations["TARGET"].sort_values().drop("TARGET")
    
    # Get the top 15 positive correlated values  
    top_positive_corr = sorted_corr[-10:]  
      
    # Get the top 15 negative correlated values  
    top_negative_corr = sorted_corr[:10]  
    
    # Create subplots for both tables  
    fig, axes = plt.subplots(1, 2, figsize=(25, 8))  
      
    # Define a color palette  
    color_palette = sns.color_palette("coolwarm", len(top_positive_corr))  
    
    axes[0].barh(top_positive_corr.index, top_positive_corr.values, color=color_palette)  
    axes[0].set_title("Top 10 correlations positives")  
      
    # Plot the top negative correlated values  
    axes[1].barh(top_negative_corr.index, top_negative_corr.values, color=color_palette[::-1])  
    axes[1].set_title("Top 10 correlations négatives")  
    
    plt.subplots_adjust(wspace=0.4)  
    plt.show()
    
    return correlations

def test_sum_living_features(dataframe):
    df = dataframe.copy()
    living_features = [ 'APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG', 'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG', 'APARTMENTS_MODE', 'BASEMENTAREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BUILD_MODE', 'COMMONAREA_MODE', 'ELEVATORS_MODE', 'ENTRANCES_MODE', 'FLOORSMAX_MODE', 'FLOORSMIN_MODE', 'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE', 'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE', 'APARTMENTS_MEDI', 'BASEMENTAREA_MEDI', 'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI', 'ELEVATORS_MEDI', 'ENTRANCES_MEDI', 'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI', 'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI', 'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI',  'TOTALAREA_MODE']
    # Compute the sum of the numerical columns  
    df['sum_living_features'] = df[living_features].sum(axis=1)
    
    # Drop the original columns  
    df.drop(columns=living_features, inplace=True)
    return df

def test_sum_flag_documents(dataframe):
    df = dataframe.copy()
    flag_documents = ['FLAG_DOCUMENT_2','FLAG_DOCUMENT_3','FLAG_DOCUMENT_4','FLAG_DOCUMENT_5','FLAG_DOCUMENT_6','FLAG_DOCUMENT_7','FLAG_DOCUMENT_8','FLAG_DOCUMENT_9','FLAG_DOCUMENT_10','FLAG_DOCUMENT_11','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13','FLAG_DOCUMENT_14','FLAG_DOCUMENT_15','FLAG_DOCUMENT_16','FLAG_DOCUMENT_17','FLAG_DOCUMENT_18','FLAG_DOCUMENT_19','FLAG_DOCUMENT_20','FLAG_DOCUMENT_21']

    # Compute the sum of the numerical columns  
    df['sum_flag_documents'] = df[flag_documents].sum(axis=1)
    
    # Drop the original columns  
    df.drop(columns=flag_documents, inplace=True)
    return df


def test_sum_flag_contacts(dataframe):
    df = dataframe.copy()
    flag_contact = ['FLAG_MOBIL','FLAG_EMP_PHONE','FLAG_WORK_PHONE','FLAG_CONT_MOBILE','FLAG_PHONE','FLAG_EMAIL']

    # Compute the sum of the numerical columns  
    df['sum_flag_contacts'] = df[flag_contact].sum(axis=1)
    
    # Drop the original columns  
    df.drop(columns=flag_contact, inplace=True)
    return df

def make_mi_scores_class(X, y):
    X = X.copy()
    X = X.fillna(X.mean())
    mi_scores = mutual_info_classif(X, y, discrete_features=False, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

def plot_mi_scores(scores, ax, title):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    ax.barh(width, scores)
    ax.set_yticks(width)  
    ax.set_yticklabels(ticks)  
    ax.set_title(title)
    
def normalize_per_col(dataframe):
    df = dataframe.copy()
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df


def normalize_all_col(dataframe):
    df = dataframe.copy()
    df_flattened = df.values.flatten().reshape(-1, 1)  
    scaler = MinMaxScaler()
    normalized_globally = scaler.fit_transform(df_flattened)  
    df_reshaped = normalized_globally.reshape(-1, len(df.columns))  
    df_normalized_globally = pd.DataFrame(df_reshaped, columns=df.columns)  

    return df_normalized_globally

def prepare_dataFrame_for_feature_engineering(d):
    data = copy.deepcopy(d)
    # SK_ID_CURR
    # Save SK_ID_CURR, it's an ID and shouldn't be used for fitting & predicting
    data["test_df_SK_ID_CURR"] = data["test_df"][['SK_ID_CURR']].copy()
    # Drop SK_ID_CURR from dataFrame
    data["df"].drop(columns=['SK_ID_CURR'], inplace=True)
    data["test_df"].drop(columns=['SK_ID_CURR'], inplace=True)

    # Set y
    data["y"] = data["df"]["TARGET"]

    # Align the dataFrames, keep only the columns present in both dataFrames
    data["df_aligned"], data["test_df_aligned"] = data["df"].align(data["test_df"], join='inner', axis=1)
    
    # Set X after aligned df
    data["X"] = data["df_aligned"]
    
    # Split the model into train and test with a test size of 20%
    data["X_train"], data["X_test"], data["y_train"], data["y_test"] = train_test_split(
        data["X"], 
        data["y"],
        test_size=0.2, 
        random_state=42
    )

    # Impute the dataFrame using the mean strategy
    simple_imputer = SimpleImputer(strategy="mean")
    data["X_train"] = pd.DataFrame(simple_imputer.fit_transform(data["X_train"]))
    data["X_train"].columns = data["X"].columns
    
    # Set test separately to avoid train-test contamination
    data["X_test"] = pd.DataFrame(simple_imputer.transform(data["X_test"]))
    data["X_test"].columns = data["X"].columns
    
    # Set the real world test model using the same imputer
    data["test_df_aligned"] = pd.DataFrame(simple_imputer.transform(data["test_df_aligned"]))
    data["test_df_aligned"].columns = data["X"].columns
    data["imputer"] = simple_imputer
    return data

def MSE(y_pred, y_real):
    return np.mean((y_pred - y_real)**2)

def prepare_submission(data, pred, name):
    submit = data["test_df_SK_ID_CURR"]
    submit['TARGET'] = pred
    submit.to_csv(f"./submissions/{name}.csv", index = False)
    
def create_baseline(data, model_name, model, data_name):
    ### Baseline for model
    elapsed_time_fit = 0
    elapsed_time_predict = 0
    
    # Fit the model
    if model_name not in data:
        start_time = timeit.default_timer()
        
        if model_name.startswith("nn_"):
            data[f"{model_name}_history"] = model.fit(data["X_train"], data["y_train"], epochs=10, batch_size=32, verbose=0)
        else:
            model.fit(data["X_train"], data["y_train"])
        elapsed_time_fit = timeit.default_timer() - start_time
        data[model_name] = model

    # Predict the model
    model_name_pred = f"{model_name}_pred"
    if model_name_pred not in data:
        start_time = timeit.default_timer()
        data[model_name_pred] = data[model_name].predict(data["X_test"])
        elapsed_time_predict = timeit.default_timer() - start_time

    mse_name = f"{model_name}_mse"
    data[mse_name] = mean_squared_error(data[model_name_pred], data['y_test'])
    
    if model_name.startswith("nn_"):
        data[f"{model_name}_test_proba"] = data[model_name].predict(data["test_df_aligned"])
    else:
        data[f"{model_name}_test_proba"] = data[model_name].predict_proba(data["test_df_aligned"])[:, 1]
        data[f"{model_name}_local_proba"] = data[model_name].predict_proba(data["X_test"])[:, 1]
    
    prepare_submission(data, data[f"{model_name}_test_proba"], f"{data_name}_{model_name}")
    print(f"Model {model_name} fit: {elapsed_time_fit:.2f}s, predict: {elapsed_time_predict:.2f}s, mse: {data[mse_name]:.4f}")
    
    
#### 
def check_feature_importances(data):
    data = {  
        'feature': data["df_aligned"].columns,  
        'importance': data["rf_class"].feature_importances_  
    }  
    
    df = pd.DataFrame(data)  
    
    sorted_df = df.sort_values(by='importance', ascending=False)
    return sorted_df

###### GRAPHS
def show_graph_local(dataframe, flat_dataframe):
    df = dataframe.copy()  
    df_long = df.melt(var_name='model', value_name='MSE')  
    sns.set(style="whitegrid", palette="colorblind")  

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 3.5), gridspec_kw={'width_ratios': [1, 1]})  
    sns.stripplot(x='model', y='MSE', hue=df_long.index, data=df_long, jitter=True, dodge=True, ax=ax1)  

    ax1.set_xlabel('MSE Model')  
    ax1.set_ylabel('MSE Score')  
    ax1.set_title('MSE Scores of Different Dataframes')  

    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)  
    ax1.legend([],[], frameon=False)

    top_5_indices = flat_dataframe['mse'].nsmallest(5).index  
    top_5_scores = flat_dataframe.loc[top_5_indices, :]  
    top_5_scores = top_5_scores.round(6)  

    table = ax2.table(cellText=top_5_scores.values,  
                    colLabels=top_5_scores.columns,  
                    loc='right'
                    , bbox=[0, 0, 1, 1])  
    table.auto_set_font_size(False)  
    table.set_fontsize(10)
    pad = 0.05  # Increase this value to adjust the width  
    ax2.axis('off')  

    plt.tight_layout()  
    plt.show()  
  
def show_graph_kaggle(dataframe, flat_dataframe):
    df = dataframe.copy()  
    df_long = df.melt(var_name='model', value_name='MSE')  
    sns.set(style="whitegrid", palette="colorblind")  

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 3.5), gridspec_kw={'width_ratios': [1, 1]})  
    sns.stripplot(x='model', y='MSE', hue=df_long.index, data=df_long, jitter=True, dodge=True, ax=ax1)  

    ax1.set_xlabel('Score Model')  
    ax1.set_ylabel('Score')  
    ax1.set_title('Scores of Different Dataframes')  

    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)  
    ax1.legend([],[], frameon=False)

    top_5_indices = flat_dataframe['score'].nlargest(5).index  
    top_5_scores = flat_dataframe.loc[top_5_indices, :]  
    top_5_scores = top_5_scores.round(6)  

    table = ax2.table(cellText=top_5_scores.values,  
                    colLabels=top_5_scores.columns,  
                    loc='right'
                    , bbox=[0, 0, 1, 1])  
    table.auto_set_font_size(False)  
    table.set_fontsize(10)
    pad = 0.05  # Increase this value to adjust the width  
    ax2.axis('off')  

    plt.tight_layout()  
    plt.show()
  
  
def get_false_negatives_false_positives(y_true, y_pred):  
    # Calculate the confusion matrix  
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()  
        
    # Return the number of false negatives and false positives  
    return fn, fp, tp, tn

def check_sample(dataFrame, sample):
    
    # Create the exlainer
    explainer = LimeTabularExplainer(
        dataFrame["X_train"].to_numpy(),
        feature_names= dataFrame["X_train"].columns,
    )
    exp = explainer.explain_instance(
        dataFrame["X_test"].iloc[sample],
        dataFrame["xgb_class_best"].predict_proba,
    )
    
    return exp

def undersample_dataframe(df, target_column):
    random_sampler = RandomUnderSampler(
        sampling_strategy=1.0,  # perfect balance
        random_state=42,
    )

    df, _ = random_sampler.fit_resample(df, df[[target_column]])
    df = df.reset_index(drop=True)
    return df

def show_feature_balance(feature):
    class_counts = feature.value_counts()  
    class_labels = ["Bon client", "Mauvais client"] #class_counts.index  
    class_values = class_counts.values

    plt.figure(figsize=(8, 2))
    sns.barplot(x=class_labels, y=class_values)  
    plt.xlabel('Type')  
    plt.ylabel('Nombre d\'individu')  
    plt.title('Balance des bons et mauvais payeurs')

    plt.show()
    
def custom_scorer(y_true, y_pred):  
    # Define the cost of false negatives and false positives  
    false_negative_cost = 10  
    false_positive_cost = 1
      
    # Calculate the number of false negatives and false positives  
    false_negatives = ((y_true == 1) & (y_pred == 0)).sum()  
    false_positives = ((y_true == 0) & (y_pred == 1)).sum()  
      
    # Calculate the total cost  
    total_cost = (false_negative_cost * false_negatives) + (false_positive_cost * false_positives)  
      
    # Return the negative of the total cost as the score  
    return -total_cost

def display_auc(model_name, ax, dfs):
    # fig, ax = plt.subplots(figsize=(10, 6))
    best_roc = 0
    best_dataframe = ""
    for i in dfs.keys():
        false_positive_rate, true_positive_rate, thresholds = roc_curve(dfs[i]["y_test"],dfs[i][f"{model_name}_pred"])  
        roc_auc = auc(false_positive_rate, true_positive_rate)
        ax.plot(false_positive_rate, true_positive_rate, label=f"{i} = %0.2f" % roc_auc)
        if roc_auc > best_roc:
            best_roc = roc_auc
            best_dataframe = i
    # ax.plot([0,1],[0,1], linestyle="--")    
    ax.set_title(f"ROC for {model_name}")    
    ax.set_xlabel("False Positive Rate")    
    ax.set_ylabel("True Positive Rate")    
    # ax.legend(loc="lower right")    
    ax.axis("tight")
    return (best_dataframe, best_roc)

def get_confusion_matrix(y_test, y_pred):
    mat = confusion_matrix(y_test, y_pred)
    mat = pd.DataFrame(mat)
    mat.columns = [f"pred_{i}" for i in mat.columns]
    mat.index = [f"test_{i}" for i in mat.index]
    return mat

def show_confusion_matrix(y_test, y_pred, axs):
     axs.set_title(f"Matrice de confusion")

     table = axs.table(cellText=get_confusion_matrix(y_test, y_pred).values,  
                          colLabels=get_confusion_matrix(y_test, y_pred).columns,  
                          cellLoc='center', rowLoc='center', fontsize=40)
     axs.axis('off') 
     for i, cell in enumerate(table._cells.values()):
          cell_text = cell.get_text()  
          cell_text.set_fontsize(10)
          
          if i == 0 or i == 3:
               cell.set_facecolor('#7fff7f')  # optional: set cell border color
               
          if i == 1 or i == 2:
               cell.set_facecolor('#ff7f7f')  # optional: set cell border color
               
def show_cost_difference_fn_fp(threshold, dataFrame):
    dataFrame['local_test'] = np.where(dataFrame['xgb_class_best_local_proba'] > threshold, 1, 0)
    fn, fp, tp, tn = get_false_negatives_false_positives(dataFrame["y_test"], dataFrame["local_test"])
    # print(tp, fp)
    # print(fn, tn)

    avg_credit = 1
    cost_fp = fp * avg_credit
    cost_fn = fn * avg_credit * 10

    # Create dataframe for costs  
    costs = pd.DataFrame({'Type': ['Faux Négatif', 'Faux Positif'],  
                        'Coût': [cost_fn, cost_fp]})  
    
    # Create bar plot  
    sns.barplot(x='Type', y='Coût', data=costs)  
    plt.xlabel('Type')  
    plt.ylabel('Coût')  
    plt.title('Coût FN vs FP')  
    plt.show()
    
def get_all_dataframe(app_train, app_test):
    df_functions = {  
    "sum_living_features": test_sum_living_features,  
    "sum_flag_documents": test_sum_flag_documents,  
    "sum_flag_contacts": test_sum_flag_contacts,
    "sum_flag_living_contact": lambda df: test_sum_flag_contacts(test_sum_flag_documents(test_sum_living_features(df)))  
    }

    dataframes = {}

    dataframes["default"] = {}  
    dataframes["default"]["df"] = app_train.copy()
    dataframes["default"]["test_df"] = app_test.copy()
    print(f"Dataframe default shape = {dataframes['default']['df'].shape}")
        
    for df_name, func in df_functions.items():  
        dataframes[df_name] = {}  
        dataframes[df_name]["df"] = func(app_train)  
        dataframes[df_name]["test_df"] = func(app_test)  
        print(f"Dataframe {df_name} shape = {dataframes[df_name]['df'].shape}")
    return dataframes

def normalize_all_dataframe(dfs):
    # Copying the data because the normalization will add more keys
    keys = dfs.copy().keys()
    # Loop trhough all and normalize them
    for data_name in keys:
        print(f"Normalizing {data_name} dataframes (per columns & all columns)...")

        # Data that won't be normalized
        target = dfs[data_name]["df"]["TARGET"].copy()
        ident = dfs[data_name]["df"]["SK_ID_CURR"].copy()
        test_ident = dfs[data_name]["test_df"]["SK_ID_CURR"].copy()

        # Per columns
        norm_col_name = f"{data_name}_norm_col"
        dfs[norm_col_name] = {}
        dfs[norm_col_name]["df"] = normalize_per_col(dfs[data_name]["df"].drop(columns=["TARGET", "SK_ID_CURR"]))
        dfs[norm_col_name]["test_df"] = normalize_per_col(dfs[data_name]["test_df"].drop(columns=["SK_ID_CURR"]))
        
        dfs[norm_col_name]["df"].insert(dfs[data_name]["df"].columns.get_loc(ident.name), ident.name, ident)
        dfs[norm_col_name]["df"].insert(dfs[data_name]["df"].columns.get_loc(target.name), target.name, target)
        dfs[norm_col_name]["test_df"].insert(dfs[data_name]["test_df"].columns.get_loc(test_ident.name), test_ident.name, test_ident)
        
        
        # All columns
        norm_all_name = f"{data_name}_norm_all"
        dfs[norm_all_name] = {}
        dfs[norm_all_name]["df"] = normalize_all_col(dfs[data_name]["df"].drop(columns=["TARGET", "SK_ID_CURR"]))
        dfs[norm_all_name]["test_df"] = normalize_all_col(dfs[data_name]["test_df"].drop(columns=["SK_ID_CURR"]))

        dfs[norm_all_name]["df"].insert(dfs[data_name]["df"].columns.get_loc(ident.name), ident.name, ident)
        dfs[norm_all_name]["df"].insert(dfs[data_name]["df"].columns.get_loc(target.name), target.name, target)
        dfs[norm_all_name]["test_df"].insert(dfs[data_name]["test_df"].columns.get_loc(test_ident.name), test_ident.name, test_ident)
    
    return dfs

def prepare_stats_results(dfs, kaggle_results):
    
    model_names = ["log_reg", "xgb_class", "rf_class",  "nn_model"]

    # Initialize empty lists for data  
    data = { 'name': [] }
    data_2d_local = { 'name': [], "mse":[] }
    data_2d_kaggle = { 'name': [], "score":[] }

    local_cols = []
    kaggle_cols = []

    # Print DataFrame
    for data_name in dfs.keys():
        
        data["name"].append(data_name)
        
        for model_name in model_names:
            mse = dfs[data_name][f"{model_name}_mse"]
            key = f"local_{model_name}"
            kkey = f"kaggle_{model_name}"
            
            if key not in data:
                local_cols.append(key)
                data[key] = []
            
            if kkey not in data:
                kaggle_cols.append(kkey)
                data[kkey] = []
            
            data[key].append(mse)
            kaggle_score = kaggle_results[kaggle_results["fileName"] == f"{data_name}_{model_name}.csv"]["privateScore"].values[0]
            data[kkey].append(kaggle_score)
            
            data_2d_local["name"].append(f"{data_name}_{model_name}")
            data_2d_local["mse"].append(mse)
            
            data_2d_kaggle["name"].append(f"{data_name}_{model_name}")
            data_2d_kaggle["score"].append(kaggle_score)

    stats = pd.DataFrame(data)
    stats = stats.reset_index(drop=True)

    data_2d_local = pd.DataFrame(data_2d_local)
    data_2d_kaggle = pd.DataFrame(data_2d_kaggle)
    
    return (stats, data_2d_local, data_2d_kaggle, local_cols, kaggle_cols)

def find_best_threshold(dataFrame):
    cost_fps = []  
    cost_fns = [] 

    thresholds = np.arange(0, 1, 0.001)
    lowest_cost = float("inf")
    best_threshold = None

    for threshold in thresholds:
        dataFrame['local_test'] = np.where(dataFrame['xgb_class_best_local_proba'] > threshold, 1, 0)  
        fn, fp, tp, tn = get_false_negatives_false_positives(dataFrame["y_test"], dataFrame["local_test"])  

        c_fp = fp
        c_fn = fn * 10
        
        cost_fps.append(c_fp)  
        cost_fns.append(c_fn)  

        if abs(c_fp - c_fn) < lowest_cost:
            lowest_cost = abs(c_fn - c_fp)
            best_threshold = threshold

    plt.plot(thresholds, cost_fps, label='FP (Bon client prédit mauvais)')  
    plt.plot(thresholds, cost_fns, label='FN (Mauvais client prédit bon)')  
    plt.xlabel('Threshold')  
    plt.ylabel('Cost')  
    plt.legend()

    # Find the index of the threshold at which the two lines cross  
    crossing_threshold_index = np.argmin(np.abs(np.array(cost_fns) - np.array(cost_fps)))  
    crossing_threshold = thresholds[crossing_threshold_index]  

    # Add the crossing point to the graph
    plt.scatter(crossing_threshold, cost_fps[crossing_threshold_index], color='red', label='Crossing Point')  
    plt.show()
    return best_threshold


# Print tree decision leaf to decide the probabilities
# Plot tree
# xgb_plot_tree(booster=dataFrame["xgb_class_best"], rankdir="LR")
# fig = plt.gcf()
# fig.set_size_inches(100, 50)