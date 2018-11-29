import csv
import santa as st
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot
import pickle
from collections import defaultdict
import random

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

def find_closest_city(current, candidates):
    candidate = None
    distance = 10000000000
    if len(candidates) > 200:
        candidates = random.sample(candidates, 200)
    for city in candidates:
        dist = st.compute_edge_cost(current, city.coord)
        if dist < distance:
            candidate = city
            distance = dist
    return candidate

class City(object):
    def __init__(self, row):
        self.id = int(row['CityId'])
        self.coord = np.array([row['X'], row['Y']])

def greedy_traversal(cluster_df, c_dict):
    cluster_dict = defaultdict(set)
    print("building lookup...")
    for row in cluster_df.iterrows():
        r_dict = row[1]
        cluster_dict[r_dict['cluster']].add(City(r_dict))
    path = []
    cur_position = np.array([0.0, 0.0])
    for cluster_id, cities in cluster_dict.items():
        print("inferring cluster ", cluster_id)
        i = 0
        while len(cities) > 0:
            next_city = find_closest_city(cur_position, cities)
            cities.remove(next_city)
            cur_position = next_city.coord
            path.append(next_city.id)
            i += 1
            if i % 1000 == 0:
                print("remaining...", len(cities))
    print("Computing Cost...")
    print(st.compute_path_cost(path, c_dict))
    return path


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
path = greedy_traversal(clustered_df, cities_dict)
print("Writing Submission")
with open("./state/greedy_path.csv", "w") as outf:
    writer = csv.writer(outf)
    writer.writerow(["Path"])
    for id in path:
        writer.writerow([id])
