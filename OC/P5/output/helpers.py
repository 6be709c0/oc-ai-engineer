from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN  
from matplotlib import colormaps

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
import numpy as np
import os
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score  
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
# HELPERS.py
# Todo Variance explainability (% of explainability > 80% ?)

from sklearn.preprocessing import MinMaxScaler  
import plotly.express as px  
from matplotlib.patches import Circle  
import matplotlib.pyplot as plt  
import seaborn as sns  
from mpl_toolkits.mplot3d import Axes3D  
import plotly.graph_objects as go

def normalize_dataframe(df):
    scaler = MinMaxScaler()
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

config_kmeans_range = range(1,30)
def get_kmeans_cluster_inertia(df):
    r_inertia = []
    for k in config_kmeans_range:
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(df)
        r_inertia.append(kmeans.inertia_)
    return r_inertia

def show_kmeans_cluster_inertia(inertia, ax):
    plt.figure(figsize=(10,3))
    ax.grid()
    ax.plot(config_kmeans_range,inertia,marker='o')
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Inertia')

def show_pca_3d(df, nb_clusters, kmean_algorithm, kmean_init):  
    # Computing PCA  
    pca = PCA(n_components=3)  
    df_pca = pca.fit_transform(df)  
      
    # Running KMeans clustering  
    kmeans = KMeans(n_clusters=nb_clusters,random_state=0, algorithm=kmean_algorithm, n_init=kmean_init)  
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
            marker=dict(  
                color='red',  
                size=5,  
                symbol='x'  
            ),  
            mode='markers',  
            name='Centroids'  
        )  
    )  
    fig.show()
import matplotlib.pyplot as plt  
import matplotlib.patches as patches

def show_pca_2d(df, nb_clusters, ax, kmean_algorithm, kmean_init):
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df)
    
    kmeans = KMeans(n_clusters=nb_clusters,random_state=0, algorithm=kmean_algorithm, n_init=kmean_init)
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
        # circle = Circle((centroid[0], centroid[1]), 0.5, color='red', alpha=0.5, fill=False)  
        circle = patches.Circle((centroid[0], centroid[1]), 0.5, color='red', alpha=0.5, fill=False)  
        # circle = patches.Circle((0.5, 0.5), 0.4, edgecolor='r', facecolor='none')  
        ax.add_patch(circle)  
      
    ax.legend(loc='upper right')  
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  
    # plt.show()  
    
def show_kmeans_2d(df, nb_clusters, ax):
    kmeans = KMeans(n_clusters=nb_clusters,random_state=0, n_init=10)
    clusters = kmeans.fit_predict(df)  

    # Assuming you have the KMeans centroids available  
    centroids = kmeans.cluster_centers_  
    labels = kmeans.labels_ 
    
    df_combined = pd.DataFrame(df)  
    df_combined['Cluster'] = labels  
    
    sns.scatterplot(data=df_combined, x=df[:, 0], y=df[:, 1], hue='Cluster',ax=ax)  
    ax.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='red', s=200)
    
    # Plotting circles around the centroids  
    for i, centroid in enumerate(centroids):  
        circle = patches.Circle((centroid[0], centroid[1]), 0.5, color='red', alpha=0.5, fill=False)  
        ax.add_patch(circle)  
      
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')    



def show_3d(df, x, y, z):  
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
    
    
def show_segmentation(df, n_cluster, kmean_algorithm="lloyd",kmean_init=10):
    fig, axs = plt.subplots(1, 2, figsize=(15, 3))
    
    inertia = get_kmeans_cluster_inertia(df)

    show_kmeans_cluster_inertia(inertia, axs[0])
    show_kmeans_2d(df, n_cluster, axs[1])
    plt.show()

    # show_pca_3d(df, n_cluster, kmean_algorithm, kmean_init)
    
def do_dbscan(X_choosen, title, eps=7, min_samples=5):

    # 1. Apply DBSCAN clustering  
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)  
    # dbscan = DBSCAN(eps=10, min_samples=5)  
    cluster_labels = dbscan.fit_predict(X_choosen)  


    # 2. Determine centroids of clusters  
    unique_labels = np.unique(cluster_labels)  
    centroids = []

    for label in unique_labels:  
        mask = cluster_labels == label  
        cluster_points = X_choosen[mask]  
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid)  

    centroids = np.array(centroids)
    colormap = cm.get_cmap("Paired", len(unique_labels))  
    colormap_array = colormap.colors  

    # Plot t-SNE scatter plot with centroids and different colors  
    plt.figure(figsize=(12, 5))  
    for i, label in enumerate(unique_labels):
        mask = cluster_labels == label  
        plt.scatter(X_choosen[mask, 0], X_choosen[mask, 1], color=colormap(i), label=f'Cluster {label}')  

    # Plot centroids with black star marker  
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', color='black', s=50)  

    plt.title(title)  
    plt.show()
    
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)  
    # Compute the average silhouette score  
    average_score = silhouette_score(X_choosen, cluster_labels)
        
    # Compute the silhouette scores for each sample  
    sample_scores = silhouette_samples(X_choosen, cluster_labels)  
    
    # Create a new dataframe with the sample scores and the corresponding cluster label  
    df_silhouette = pd.DataFrame({'SilhouetteScore': sample_scores, 'Cluster': cluster_labels})  
    
    # Plot the silhouette graph  
    fig, ax = plt.subplots(figsize=(8, 6))  
    sns.boxplot(x='Cluster', y='SilhouetteScore', data=df_silhouette, ax=ax, palette=colormap_array)  
    sns.stripplot(x='Cluster', y='SilhouetteScore', data=df_silhouette, ax=ax, size=4, color="gray", edgecolor="gray", alpha=0.7)  
    ax.set_xlabel('Cluster')  
    ax.set_ylabel('Silhouette Score')  
    ax.axhline(y=average_score, color="red", linestyle="--")  
    ax.set_title('Silhouette scores of clusters')  
    
    print("n_clusters", n_clusters)
    # Display the plot  
    plt.show()
    
    return cluster_labels

def undersample_dataframe(df, target_column):
    random_sampler = RandomUnderSampler(
        sampling_strategy=1.0,  # perfect balance
        random_state=42,
    )

    df, _ = random_sampler.fit_resample(df, df[[target_column]])
    df = df.reset_index(drop=True)
    return df

def label_encoder(df):
    le = LabelEncoder()  
    df_encoded = df.copy()

    for col in df_encoded:  
        if df_encoded[col].dtype == 'object':  
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            
    return df_encoded


def undersample_dataframe(df, target_column):
    random_sampler = RandomUnderSampler(
        sampling_strategy=1.0,  # perfect balance
        random_state=42,
    )

    df, _ = random_sampler.fit_resample(df, df[[target_column]])
    df = df.reset_index(drop=True)
    return df


def calculate_silhouette_score_kmeans(dataframe, n_clusters): 
    
    df = dataframe.copy()
     
    # Fit Kmeans model  
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto")  
    kmeans.fit(df) 
  
    # Predict cluster labels  
    labels = kmeans.predict(df)  
    
    # Calculate silhouette score  
    silhouette = silhouette_score(df, labels)  
    calinski_harabasz = calinski_harabasz_score(df, labels)  
    davies_bouldin = davies_bouldin_score(df, labels)
    
    return (n_clusters, silhouette,calinski_harabasz,davies_bouldin, kmeans)  

def calculate_silhouette_score_dbscan(dataframe, eps=0.5, min_samples=10):  
    
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
    
    if(nb_clusters > 1):
        silhouette = silhouette_score(dataframe, labels)  
        calinski_harabasz = calinski_harabasz_score(dataframe, labels)  
        davies_bouldin = davies_bouldin_score(dataframe, labels) 
    # Exclude noise points (clusters with label -1)  
    # valid_clusters = df[clusters != -1]  
    # valid_labels = df[clusters != -1]  
      
    # Calculate silhouette score  
    return (nb_clusters, silhouette, calinski_harabasz, davies_bouldin, dbscan)  

def plot_radars(data, group):

    scaler = MinMaxScaler()
    data = pd.DataFrame(scaler.fit_transform(data), 
                        index=data.index,
                        columns=data.columns).reset_index()
    
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
        )),
        showlegend=True,
        title={
            'text': "Comparaison des moyennes par variable des clusters",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        title_font_color="blue",
        title_font_size=18)

    fig.show()