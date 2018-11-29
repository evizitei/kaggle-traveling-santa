import santa as st
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot
import pickle

CLUSTER_FN = "./state/clustered_cities.csv"

cities_df = pd.read_csv("all/cities.csv")
cities_dict = cities_df[['X','Y']].values

def visualize(c_df):
    pX = c_df['X'].tolist()
    pY = c_df['Y'].tolist()
    pS = [0.02 for n in range(len(c_df))]
    pyplot.scatter(pX, pY, marker=",", s=pS)
    pyplot.show()

def visualize_clusters(c_df, clusters):
    pX = c_df['X'].tolist()
    pY = c_df['Y'].tolist()
    pS = [0.02 for n in range(len(c_df))]
    pyplot.scatter(pX, pY, marker=',', s=pS, c=clusters)
    pyplot.show()

def naive_traversal(c_dict):
    # and the naive traversal will just be the ordered list of city IDs:
    path = np.arange(len(c_dict))
    return st.compute_path_cost(path, c_dict)

def sorted_traversal(c_df):
    c_dict = c_df[['X','Y']].values
    sorted_cities = c_df.sort_values(['X', 'Y'])
    sorted_path = sorted_cities.CityId.astype(int).tolist()
    return st.compute_path_cost(sorted_path, c_dict)

def cluster_cities(c_df, fn):
    cluster_labels = st.cluster_points(c_df)
    c_df['cluster'] = np.array(cluster_labels)
    c_df.to_csv(fn)


#visualize(cities_df)
#print("NAIVE: ", naive_traversal(cities_dict))  # 443,431,633
#print("SORTED: ", sorted_traversal(cities_df))   # 194,711,784
#cluster_cities(cities_df, CLUSTER_FN)
#visualize_clusters(cities_df, cluster_labels)
clustered_df = pd.read_csv(CLUSTER_FN)
print(clustered_df.head())
