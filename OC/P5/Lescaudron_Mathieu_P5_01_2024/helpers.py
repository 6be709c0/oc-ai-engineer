import warnings  
import os  
import os.path  
import pickle  
  
from datetime import datetime  
from datetime import timedelta  
  
import numpy as np  
import pandas as pd
import seaborn as sns

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib.patches as patches
from matplotlib.patches import Circle  

import plotly.express as px
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D  
  
from sklearn.cluster import DBSCAN  
from sklearn.cluster import KMeans  
from sklearn.cluster import AgglomerativeClustering

from sklearn.decomposition import PCA  
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler  
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score 
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score 
from sklearn.metrics import adjusted_rand_score  
from sklearn.cluster import KMeans

from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import SilhouetteVisualizer  
  
from imblearn.under_sampling import RandomUnderSampler  


def normalize_dataframe(df):  
    """  
    Normalizes a dataframe using a MinMaxScaler.  
      
    Args:  
    - df: The dataframe to be normalized.  
      
    Returns:  
    - The normalized dataframe.  
    """  
    scaler = MinMaxScaler()  
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)  

config_kmeans_range = range(1, 30)

def get_kmeans_cluster_inertia(df):  
    """  
    Calculates the inertia values for different numbers of 
    clusters using Kmeans clustering.  
      
    Args:  
    - df: The input dataframe.  
      
    Returns:  
    - The list of inertia values.  
    """  
    r_inertia = []  
    for k in config_kmeans_range:  
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(df)  
        r_inertia.append(kmeans.inertia_)  
    return r_inertia  
  
def show_kmeans_cluster_inertia(inertia, ax):  
    """  
    Plots the inertia values for different numbers of clusters.  
      
    Args:  
    - inertia: The list of inertia values.  
    - ax: The subplot object to plot the graph on.  
      
    Returns:  
    - None.  
    """  
    ax.grid()  
    ax.plot(config_kmeans_range, inertia, marker='o')  
    ax.set_xlabel('Number of Clusters')  
    ax.set_ylabel('Inertia')  
    
    
def show_pca_3d(df, nb_clusters, kmean_algorithm, kmean_init):  
    """  
    Visualizes the clusters in a 3D space using PCA and K-means clustering.  
      
    Args:  
    - df: The input dataframe.  
    - nb_clusters: The number of clusters.  
    - kmean_algorithm: The algorithm for K-means clustering.  
    - kmean_init: The number of initializations for K-means clustering.  
      
    Returns:  
    - None.  
    """  
    # Computing PCA  
    pca = PCA(n_components=3)  
    df_pca = pca.fit_transform(df)  
      
    # Running KMeans clustering  
    kmeans = KMeans(n_clusters=nb_clusters, random_state=0, algorithm=kmean_algorithm, n_init=kmean_init)  
    clusters = kmeans.fit_predict(df_pca)  
      
    # Extracting centroids and labels  
    centroids = kmeans.cluster_centers_  
    labels = kmeans.labels_  
      
    # Combining PCA results with cluster labels  
    df_combined = pd.DataFrame(df_pca, columns=['Component 1', 'Component 2', 'Component 3'])  
    df_combined['Cluster'] = labels  
      
    # Plotting 3D scatter plot  
    fig = px.scatter_3d(df_combined, x='Component 1', y='Component 2', z='Component 3', color='Cluster')  
      
    # Adding centroids as scatter plot markers  
    fig.add_trace(  
        go.Scatter3d(  
            x=centroids[:, 0],  
            y=centroids[:, 1],  
            z=centroids[:, 2],  
            marker=dict(color='red', size=5, symbol='x'),  
            mode='markers',  
            name='Centroids'  
        )  
    )  
    fig.show()  


def show_pca_2d(df, nb_clusters, ax, kmean_algorithm, kmean_init):  
    """  
    Visualizes the clusters in a 2D space using PCA and K-means clustering.  
      
    Args:  
    - df: The input dataframe.  
    - nb_clusters: The number of clusters.  
    - ax: The subplot object to plot the graph on.  
    - kmean_algorithm: The algorithm for K-means clustering.  
    - kmean_init: The number of initializations for K-means clustering.  
      
    Returns:  
    - None.  
    """  
    pca = PCA(n_components=2)  
    df_pca = pca.fit_transform(df)  
      
    kmeans = KMeans(n_clusters=nb_clusters, random_state=0, algorithm=kmean_algorithm, n_init=kmean_init)  
    clusters = kmeans.fit_predict(df_pca)  
  
    # Assuming you have the KMeans centroids available  
    centroids = kmeans.cluster_centers_  
    labels = kmeans.labels_  
  
    df_combined = pd.DataFrame(df_pca, columns=['Component 1', 'Component 2'])  
    df_combined['Cluster'] = labels  
  
    sns.scatterplot(data=df_combined, x='Component 1', y='Component 2', hue='Cluster', ax=ax)  
    ax.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='red', s=200)  
  
    # Plotting circles around the centroids  
    for i, centroid in enumerate(centroids):  
        circle = patches.Circle((centroid[0], centroid[1]), 0.5, color='red', alpha=0.5, fill=False)  
        ax.add_patch(circle)  
  
    ax.legend(loc='upper right')  
  
def show_kmeans_2d(df, nb_clusters, ax):  
    """  
    Visualizes the clusters in a 2D space using K-means clustering.  
      
    Args:  
    - df: The input dataframe.  
    - nb_clusters: The number of clusters.  
    - ax: The subplot object to plot the graph on.  
      
    Returns:  
    - None.  
    """  
    kmeans = KMeans(n_clusters=nb_clusters, random_state=0, n_init=10)  
    clusters = kmeans.fit_predict(df)  
  
    # Assuming you have the KMeans centroids available  
    centroids = kmeans.cluster_centers_  
    labels = kmeans.labels_  
  
    df_combined = pd.DataFrame(df)  
    df_combined['Cluster'] = labels  
  
    sns.scatterplot(data=df_combined, x=df[:, 0], y=df[:, 1], hue='Cluster', ax=ax)  
    ax.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='red', s=200)  
  
    # Plotting circles around the centroids  
    for i, centroid in enumerate(centroids):  
        circle = patches.Circle((centroid[0], centroid[1]), 0.5, color='red', alpha=0.5, fill=False)  
        ax.add_patch(circle)  
  
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  
  
def show_3d(df, x, y, z):  
    """  
    Visualizes a dataframe in a 3D scatter plot.  
      
    Args:  
    - df: The input dataframe.  
    - x: The column name for X-axis.  
    - y: The column name for Y-axis.  
    - z: The column name for Z-axis.  
      
    Returns:  
    - None.  
    """  
    # Plotting 3D scatter plot  
    fig = px.scatter_3d(  
        df,  
        x=df[x],  
        y=df[y],  
        z=df[z],  
        title="RFM",  
        opacity=0.5,  
        width=1200,  
        height=800,  
    )  
      
    fig.show()  
  
def show_segmentation(df, n_cluster, kmean_algorithm="lloyd", kmean_init=10):  
    """  
    Shows the segmentation of a dataframe using K-means clustering and various visualization techniques.  
      
    Args:  
    - df: The input dataframe.  
    - n_cluster: The number of clusters.  
    - kmean_algorithm: The algorithm for K-means clustering.  
    - kmean_init: The number of initializations for K-means clustering.  
      
    Returns:  
    - None.  
    """  
    fig, axs = plt.subplots(1, 2, figsize=(15, 3))  
      
    inertia = get_kmeans_cluster_inertia(df)  
  
    show_kmeans_cluster_inertia(inertia, axs[0])  
    show_kmeans_2d(df, n_cluster, axs[1])  
    plt.show()
    
def do_dbscan(X_chosen, title, eps=7, min_samples=5):  
    """  
    Apply DBSCAN clustering and plot t-SNE scatter plot with centroids and different colors.  
    
    Args:  
    - X_chosen: The chosen data for clustering.  
    - title: The title of the plot.  
    - eps: The maximum distance between two samples for them to be considered as in the same neighborhood.  
    - min_samples: The number of samples in a neighborhood for a point to be considered as a core point.  
    
    Returns:  
    - cluster_labels: The cluster labels assigned by DBSCAN.  
    """  
    # Apply DBSCAN clustering  
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)  
    cluster_labels = dbscan.fit_predict(X_chosen)  
  
    # Determine centroids of clusters  
    unique_labels = np.unique(cluster_labels)  
    centroids = []  
  
    for label in unique_labels:  
        mask = cluster_labels == label  
        cluster_points = X_chosen[mask]  
        centroid = np.mean(cluster_points, axis=0)  
        centroids.append(centroid)  
  
    centroids = np.array(centroids)  
    colormap = cm.get_cmap("Paired", len(unique_labels))  
    colormap_array = colormap.colors  
  
    # Plot t-SNE scatter plot with centroids and different colors  
    plt.figure(figsize=(12, 5))  
    for i, label in enumerate(unique_labels):  
        mask = cluster_labels == label  
        plt.scatter(X_chosen[mask, 0], X_chosen[mask, 1], color=colormap(i), label=f'Cluster {label}')  
  
    # Plot centroids with black star marker  
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', color='black', s=50)  
  
    plt.title(title)  
    plt.show()  
  
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)  
    # Compute the average silhouette score  
    average_score = silhouette_score(X_chosen, cluster_labels)  
  
    # Compute the silhouette scores for each sample  
    sample_scores = silhouette_samples(X_chosen, cluster_labels)  
  
    # Create a new dataframe with the sample scores and the corresponding cluster label  
    df_silhouette = pd.DataFrame({'SilhouetteScore': sample_scores, 'Cluster': cluster_labels})  
  
    # Plot the silhouette graph  
    fig, ax = plt.subplots(figsize=(8, 6))  
    sns.boxplot(x='Cluster', y='SilhouetteScore', data=df_silhouette, ax=ax, palette=colormap_array)  
    sns.stripplot(x='Cluster', y='SilhouetteScore', data=df_silhouette, ax=ax, size=4, color="gray",  
                  edgecolor="gray", alpha=0.7)  
    ax.set_xlabel('Cluster')  
    ax.set_ylabel('Silhouette Score')  
    ax.axhline(y=average_score, color="red", linestyle="--")  
    ax.set_title('Silhouette scores of clusters')  
  
    print("n_clusters", n_clusters)  
    # Display the plot  
    plt.show()  
  
    return cluster_labels  
  
  
def undersample_dataframe(df, target_column):  
    """  
    Undersample the dataframe for a balanced dataset with the given target column  
  
    Args:  
    - df: The original dataframe.  
    - target_column: The target column for balancing the dataset.  
  
    Returns:  
    - df: The undersampled dataframe.  
    """  
    random_sampler = RandomUnderSampler(  
        sampling_strategy=1.0,  # perfect balance  
        random_state=42,  
    )  
  
    df, _ = random_sampler.fit_resample(df, df[[target_column]])  
    df = df.reset_index(drop=True)  
    return df  
  
  
def label_encoder(df):  
    """  
    Label encode the categorical columns in the dataframe.  
  
    Args:  
    - df: The original dataframe.  
  
    Returns:  
    - df_encoded: The label encoded dataframe.  
    """  
    le = LabelEncoder()  
    df_encoded = df.copy()  
  
    for col in df_encoded:  
        if df_encoded[col].dtype == 'object':  
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))  
  
    return df_encoded  
def calculate_score_kmeans(dataframe, n_clusters, algorithm):  
    """  
    Calculate and return the silhouette, Calinski-Harabasz, Davies-Bouldin scores, and KMeans model   
    for a given number of clusters using the K-Means algorithm.  
      
    Args:  
    - dataframe: The input dataframe.  
    - n_clusters: The number of clusters.  
    - algorithm: The algorithm to use for the K-Means model.  
      
    Returns:  
    - A tuple containing the number of clusters, silhouette score, Calinski-Harabasz score,   
      Davies-Bouldin score, and KMeans model.  
    """  
    df = dataframe.copy()  
      
    # Fit KMeans model  
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', algorithm=algorithm)  
    kmeans.fit(df)  
      
    # Predict cluster labels  
    labels = kmeans.predict(df)  
      
    # Calculate silhouette score  
    silhouette = silhouette_score(df, labels)  
    calinski_harabasz = calinski_harabasz_score(df, labels)  
    davies_bouldin = davies_bouldin_score(df, labels)  
      
    return (n_clusters, silhouette, calinski_harabasz, davies_bouldin, kmeans)  
  
def calculate_score_agglomerative(dataframe, n_clusters, linkage):  
    """  
    Calculate and return the silhouette, Calinski-Harabasz, Davies-Bouldin scores, and AgglomerativeClustering model   
    for a given number of clusters using the Agglomerative Clustering algorithm.  
      
    Args:  
    - dataframe: The input dataframe.  
    - n_clusters: The number of clusters.  
    - linkage: The linkage criterion to use for the Agglomerative Clustering model.  
      
    Returns:  
    - A tuple containing the number of clusters, silhouette score, Calinski-Harabasz score,   
      Davies-Bouldin score, and AgglomerativeClustering model.  
    """  
    df = dataframe.copy()  
      
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)  
    agglomerative_labels = agglomerative.fit_predict(df)  
      
    # Calculate silhouette score  
    silhouette = silhouette_score(df, agglomerative_labels)  
    calinski_harabasz = calinski_harabasz_score(df, agglomerative_labels)  
    davies_bouldin = davies_bouldin_score(df, agglomerative_labels)  
      
    return (n_clusters, silhouette, calinski_harabasz, davies_bouldin, agglomerative)  
  
def calculate_score_gaussian_mixture(dataframe, n_components, covariance_type):  
    """  
    Calculate and return the silhouette, Calinski-Harabasz, Davies-Bouldin scores, and Gaussian Mixture model   
    for a given number of components using the Gaussian Mixture algorithm.  
      
    Args:  
    - dataframe: The input dataframe.  
    - n_components: The number of components.  
    - covariance_type: The covariance type to use for the Gaussian Mixture model.  
      
    Returns:  
    - A tuple containing the number of components, silhouette score, Calinski-Harabasz score,   
      Davies-Bouldin score, and Gaussian Mixture model.  
    """  
    df = dataframe.copy()  
      
    model = GaussianMixture(n_components=n_components, covariance_type=covariance_type)  
    labels = model.fit_predict(df)  
      
    # Calculate silhouette score  
    silhouette = silhouette_score(df, labels)  
    calinski_harabasz = calinski_harabasz_score(df, labels)  
    davies_bouldin = davies_bouldin_score(df, labels)  
      
    return (n_components, silhouette, calinski_harabasz, davies_bouldin, model)  
  
def calculate_score_dbscan(dataframe, eps=0.5, min_samples=10):  
    """  
    Calculate and return the number of clusters, silhouette, Calinski-Harabasz, Davies-Bouldin scores, and DBSCAN model   
    for a given input dataframe using the DBSCAN algorithm.  
      
    Args:  
    - dataframe: The input dataframe.  
    - eps: The maximum distance between two samples for them to be considered as in the same neighborhood.  
    - min_samples: The number of samples in a neighborhood for a point to be considered as a core point.  
      
    Returns:  
    - A tuple containing the number of clusters, silhouette score, Calinski-Harabasz score, Davies-Bouldin score,  
      and DBSCAN model.  
    """  
    df = dataframe.copy()  
      
    # Initialize DBSCAN model  
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)  
      
    # Fit DBSCAN model to the data  
    labels = dbscan.fit_predict(df)  
    unique_labels = np.unique(labels)  
      
    nb_clusters = len(unique_labels)  
    silhouette = 0  
    calinski_harabasz = 0  
    davies_bouldin = 0  
      
    if nb_clusters > 1:  
        silhouette = silhouette_score(dataframe, labels)  
        calinski_harabasz = calinski_harabasz_score(dataframe, labels)  
        davies_bouldin = davies_bouldin_score(dataframe, labels)  
      
    return (nb_clusters, silhouette, calinski_harabasz, davies_bouldin, dbscan)


def plot_radars(data, group):  
    """    
    Plots radar charts for each cluster in the given data.    
        
    Args:    
    - data: The data to plot radar charts for.    
    - group: The column name representing the clusters/groups.    
        
    Returns:    
    - None.    
    """  
    scaler = MinMaxScaler()  
    data = pd.DataFrame(  
        scaler.fit_transform(data),   
        index=data.index,  
        columns=data.columns  
    ).reset_index()  
      
    fig = go.Figure()  
  
    for k in data[group]:  
        fig.add_trace(go.Scatterpolar(  
            r=data[data[group]==k].iloc[:,1:].values.reshape(-1),  
            theta=data.columns[1:],  
            fill='toself',  
            name='Cluster '+str(k)  
        ))
  
    fig.update_layout(  
        polar=dict(  
            radialaxis=dict(  
                visible=True,  
                range=[0, 1]  
            )  
        ),  
        showlegend=True,  
        title={  
            'text': "Comparaison des moyennes par variable des clusters",  
            'y':0.95,  
            'x':0.5,  
            'xanchor': 'center',  
            'yanchor': 'top'  
        },  
        title_font_color="blue",  
        title_font_size=18  
    )  
  
    fig.show()  
      
def get_file_data(data, name):  
    """    
    Gets the data for the specified file.    
        
    Args:    
    - data: The dataframe to get the data from.    
    - name: The name of the file to get the data for.    
        
    Returns:    
    - The DataFrame containing the data for the specified file.    
    """  
    print(data[name].shape)  
    print(data[name].info())  
    return data[name].head()  


def get_segmented_data(end_date):
    """  
    Retrieves and processes data to create segmented dataset.  
      
    Args:  
    - end_date: The end date to filter the data.  
      
    Returns:  
    - final_df: The final dataset.  
    - undersample_df: The undersampled dataset.  
    - df_half: The half dataset.  
    """
    # Get datafiles
    pd.DataFrame(os.listdir("../input/"), columns=["Nom des fichiers"])
    directory = "../input/"  
  
    file_names = os.listdir(directory)  
    data = {}

    for file_name in file_names:  
        data[file_name] = pd.read_csv(os.path.join(directory, file_name))
        
    data["olist_orders_dataset.csv"]["order_purchase_timestamp"] = pd.to_datetime(data["olist_orders_dataset.csv"]["order_purchase_timestamp"])
    data["olist_orders_dataset.csv"] = data["olist_orders_dataset.csv"][data["olist_orders_dataset.csv"]["order_purchase_timestamp"] <= end_date]
    
    clients = data["olist_customers_dataset.csv"]
    
    clients_count = clients.groupby('customer_unique_id').size().reset_index(name='nb_orders')
    clients_count = clients_count.sort_values(by='nb_orders', ascending=False)
    clients_count.head()
    
    merged_orders = clients.merge(data["olist_orders_dataset.csv"], on="customer_id")  
    orders_reviews = data["olist_order_reviews_dataset.csv"].copy()
    # Sort to keep the most recent
    orders_reviews = orders_reviews.sort_values('review_answer_timestamp', ascending=False)  
    orders_reviews = orders_reviews.drop_duplicates('order_id')  
    orders_reviews = orders_reviews.reset_index(drop=True)  
    
    merged_orders = merged_orders.merge(orders_reviews[orders_reviews["order_id"].isin(merged_orders["order_id"])], on="order_id", how="outer")  
    merged_orders["review_score"].fillna(-1, inplace=True)
    merged_orders["review_score"].isna().sum()
    
    df = data["olist_orders_dataset.csv"].copy()

    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])  # Group by year and month  
    df['year_month'] = df['order_purchase_timestamp'].dt.to_period('M')
    
    merged_orders["order_purchase_timestamp"] = pd.to_datetime(merged_orders["order_purchase_timestamp"])  
    merged_orders["order_approved_at"] = pd.to_datetime(merged_orders["order_approved_at"])  
    merged_orders["order_delivered_customer_date"] = pd.to_datetime(merged_orders["order_delivered_customer_date"])  
    merged_orders["order_delivered_carrier_date"] = pd.to_datetime(merged_orders["order_delivered_carrier_date"])  
    merged_orders["order_estimated_delivery_date"] = pd.to_datetime(merged_orders["order_estimated_delivery_date"])  
    merged_orders["review_creation_date"] = pd.to_datetime(merged_orders["review_creation_date"])  
    merged_orders["review_answer_timestamp"] = pd.to_datetime(merged_orders["review_answer_timestamp"])  

    merged_orders["delivery_delay"] = (merged_orders["order_delivered_customer_date"] - merged_orders["order_purchase_timestamp"]).dt.days
    merged_orders["delivery_delay"].fillna(-1,inplace=True)
    
    warnings.simplefilter("ignore")  
    data["product_category_name_translation.csv"] = data["product_category_name_translation.csv"].append({"product_category_name": "undefined", "product_category_name_english":"undefined"}, ignore_index=True)
    warnings.filterwarnings('default')  

    data["olist_products_dataset.csv"]["product_category_name"].fillna("undefined", inplace=True)
    
    item_dataset = data["olist_order_items_dataset.csv"].merge(data["olist_products_dataset.csv"], on="product_id")
    item_dataset = item_dataset.merge(data["product_category_name_translation.csv"], on="product_category_name")

    categories = {  
    'agro_industry_and_commerce': 'Home & Furniture',  
    'air_conditioning': 'Electronics & Appliances',  
    'art': 'Miscellaneous',  
    'arts_and_craftmanship': 'Miscellaneous',  
    'audio': 'Electronics & Appliances',  
    'auto': 'Auto',  
    'baby': 'Toys & Baby',  
    'bed_bath_table': 'Home & Furniture',  
    'books_general_interest': 'Books',  
    'books_imported': 'Books',  
    'books_technical': 'Books',  
    'cds_dvds_musicals': 'Music',  
    'christmas_supplies': 'Miscellaneous',  
    'cine_photo': 'Electronics & Appliances',  
    'computers': 'Electronics & Appliances',  
    'computers_accessories': 'Electronics & Appliances',  
    'consoles_games': 'Miscellaneous',  
    'construction_tools_construction': 'Tools',  
    'construction_tools_lights': 'Tools',  
    'construction_tools_safety': 'Tools',  
    'cool_stuff': 'Miscellaneous',  
    'undefined': 'Miscellaneous',  
    'costruction_tools_garden': 'Tools',  
    'costruction_tools_tools': 'Tools',  
    'diapers_and_hygiene': 'Toys & Baby',  
    'drinks': 'Food and Drinks',  
    'dvds_blu_ray': 'Electronics & Appliances',  
    'electronics': 'Electronics & Appliances',  
    'fashio_female_clothing': 'Fashion',  
    'fashion_bags_accessories': 'Fashion',  
    'fashion_childrens_clothes': 'Fashion',  
    'fashion_male_clothing': 'Fashion',  
    'fashion_shoes': 'Fashion',  
    'fashion_sport': 'Sports',  
    'fashion_underwear_beach': 'Fashion',  
    'fixed_telephony': 'Electronics & Appliances',  
    'flowers': 'Home & Furniture',  
    'food': 'Food and Drinks',  
    'food_drink': 'Food and Drinks',  
    'furniture_bedroom': 'Home & Furniture',  
    'furniture_decor': 'Home & Furniture',  
    'furniture_living_room': 'Home & Furniture',  
    'furniture_mattress_and_upholstery': 'Home & Furniture',  
    'garden_tools': 'Tools',  
    'health_beauty': 'Health and Beauty',  
    'home_appliances': 'Electronics & Appliances',  
    'home_appliances_2': 'Electronics & Appliances',  
    'home_comfort_2': 'Home & Furniture',  
    'home_confort': 'Home & Furniture',  
    'home_construction': 'Home & Furniture',  
    'housewares': 'Home & Furniture',  
    'industry_commerce_and_business': 'Home & Furniture',  
    'kitchen_dining_laundry_garden_furniture': 'Home & Furniture',  
    'la_cuisine': 'Home & Furniture',  
    'luggage_accessories': 'Fashion',  
    'market_place': 'Miscellaneous',  
    'music': 'Music',  
    'musical_instruments': 'Music',  
    'office_furniture': 'Home & Furniture',  
    'party_supplies': 'Miscellaneous',  
    'perfumery': 'Health and Beauty',  
    'pet_shop': 'Pets',  
    'security_and_services': 'Electronics & Appliances',  
    'signaling_and_security': 'Electronics & Appliances',  
    'small_appliances': 'Electronics & Appliances',  
    'small_appliances_home_oven_and_coffee': 'Electronics & Appliances',  
    'sports_leisure': 'Sports',  
    'stationery': 'Home & Furniture',  
    'tablets_printing_image': 'Electronics & Appliances',  
    'telephony': 'Electronics & Appliances',  
    'toys': 'Toys & Baby',  
    'watches_gifts': 'Fashion'  
    }    

    item_dataset['category'] = item_dataset['product_category_name_english'].map(categories)
    item_dataset_encoded = pd.get_dummies(item_dataset['category'], prefix='cat')  
    item_dataset_encoded = pd.concat([item_dataset, item_dataset_encoded], axis=1)  

    feature_cat_columns = [col for col in item_dataset_encoded.columns if col.startswith('cat_')]  

    agg_dict = {'items_nb': ('order_item_id', 'count'),  
                'sum_price': ('price', 'sum'),  
                'sum_freight_value': ('freight_value', 'sum')}  
    
    for col in feature_cat_columns:  
        agg_dict[col] = (col, 'max')  
    order_items_agg = item_dataset_encoded.groupby('order_id').agg(**agg_dict).reset_index()  


    merged_orders = merged_orders.merge(order_items_agg, on="order_id", how='outer')  
    merged_orders["sum_price"].fillna(0, inplace=True)
    merged_orders["items_nb"].fillna(0, inplace=True)
    
    agg_dict = {'avg_delivery_delay': ('delivery_delay', 'mean'),  
            'items_nb': ('items_nb', 'sum'),  
            'orders_nb': ('order_id', 'count'),  
            'last_order_date': ('order_purchase_timestamp', 'max'),  
            'avg_satisfaction': ('review_score', 'mean'),  
            'sum_price': ('sum_price', 'sum'),  
            }  
  
    # Add each feature column to the aggregation dictionary with 'first' as the aggregation function  
    for col in feature_cat_columns:  
        agg_dict[col] = (col, 'sum')  
    
    # Apply the aggregation operation  
    df_pivot = merged_orders.groupby('customer_unique_id').agg(**agg_dict).reset_index()  
    
    # Most recent purchase
    most_recent_purchase = df_pivot["last_order_date"].max()

    # Add a new column to know the last purchase date
    df_pivot["days_since_last_purchase"] = round((df_pivot["last_order_date"] - most_recent_purchase) / np.timedelta64(1, "D"))
    
    final_df = df_pivot.copy()
    final_df["recency"] = final_df["days_since_last_purchase"]
    final_df["frequency"] = final_df["orders_nb"]
    final_df["monetary"] = final_df["sum_price"]

    final_df.drop(columns=["days_since_last_purchase","orders_nb","sum_price" ], axis=1, inplace=True)
    
    # Calculate the quintiles for recency  
    final_df['recency_score'] = pd.qcut(final_df['recency'], q=5, labels=False) + 1

    final_df['frequency_rank'] = final_df['frequency'].rank(method='min')
    # Calculate the quintiles for frequency  
    final_df['frequency_score'] = pd.cut(final_df['frequency_rank'], bins=5, labels=False, include_lowest=True) + 1  
    final_df.drop(['frequency_rank'], axis=1, inplace=True)
    # Calculate the quintiles for monetary  
    final_df['monetary_score'] = pd.qcut(final_df['monetary'], q=5, labels=False) + 1  
    
    final_df["rfm_score"] = final_df["recency_score"].astype(str) + final_df["frequency_score"].astype(str) +final_df["monetary_score"].astype(str)
    final_df["above_one_command"] = final_df["frequency_score"] > 1


    above_one_c = final_df[final_df['above_one_command'] == True].sample(frac=0.5)  
    below_one_c = final_df[final_df['above_one_command'] == False].sample(frac=0.5)

    df_half = pd.concat([above_one_c, below_one_c])
    df_half.head()
    
    undersample_df = undersample_dataframe(final_df, "above_one_command")
    undersample_df[["recency","frequency","monetary", "avg_satisfaction"]].to_csv("rfm_review_undersample.csv", index=False)
    undersample_df[["recency_score","frequency_score","monetary_score", "avg_satisfaction"]].to_csv("rfm_review_score_undersample.csv", index=False)
    
    final_df.drop(columns=["above_one_command", "recency","frequency","monetary"], axis=1, inplace=True)
    undersample_df.drop(columns=["above_one_command", "recency","frequency","monetary"], axis=1, inplace=True)
    df_half.drop(columns=["above_one_command", "recency","frequency","monetary"], axis=1, inplace=True)
    
    return (final_df, undersample_df, df_half)
    # undersample_df.to_csv("full_data_undersample_test.csv", index=False)
    # final_df.to_csv("full_data_test.csv", index=False)
    # df_half.to_csv("full_data_half.csv", index=False)
    
def get_original_label_of_dataframe(df, kmeans):  
    """    
    Returns the labels of the clusters assigned to the original dataframe.    
        
    Args:    
    - df: The original dataframe.    
        
    Returns:    
    - The PCA model used for dimensionality reduction.    
    - The KMeans model used for clustering.    
    - The predicted cluster labels.    
    """    
    pca = PCA(n_components=2)  
    df_pca = pca.fit_transform(df)  
  
    kmeans.fit(df_pca)  
  
    # Predict cluster labels    
    labels = kmeans.predict(df_pca)    
  
    return pca, kmeans, labels  

def get_dataframe_with_label_from_date(df_data, pca, kmeans, end_date, feature_name):  
    """    
    Returns the dataframe merged with the cluster labels based on the given end date.    
        
    Args:    
    - df_data: The original dataframe.    
    - pca: The PCA model used for dimensionality reduction.    
    - kmeans: The KMeans model used for clustering.    
    - end_date: The end date for segmenting the data.    
    - feature_name: The name of the feature to fill with cluster labels.    
        
    Returns:    
    - The modified dataframe merged with cluster labels.    
    """    
    (final_df, undersample_df, df_half) = get_segmented_data(end_date)  
    df = final_df.drop(columns=["customer_unique_id", "last_order_date"], axis=1)  
  
    df_pca = pca.transform(df)    
    labels = kmeans.predict(df_pca)  
  
    final_df["cluster"] = labels  
    final_df = final_df[["customer_unique_id", "cluster"]]  
  
    # Merge the two dataframes based on customer_unique_id    
    df_merged = df_data.merge(final_df, on='customer_unique_id', how='left')  
      
    # If no cluster, then fill na with T0  
    df_merged[feature_name] = df_merged['cluster'].fillna(df_merged['T0'])  
      
    # If no T0 because of new customer, set T0 to new feature  
    # df_merged["T0"] = df_merged['cluster'].fillna(df_merged[feature_name])  
      
    # Drop the unnecessary columns  
    df_merged = df_merged.drop(['cluster'], axis=1)    
    # Assign the modified dataframe back to df_data    
    df_data = df_merged    
  
    return df_data  

def save_model(model_filename, model):  
    """  
    Saves the model to a file using pickle.  
  
    Args:  
    - model_filename: The filename to save the model to.  
    - model: The model object to be saved.  
  
    Returns:  
    - None.  
    """  
    print("Saving...")  
    with open(model_filename, 'wb') as file:  
        pickle.dump(model, file)  