import numpy as np
import pandas as pd

from sklearn.cluster import KMeans

def cluster_points(c_df):
    pX = c_df['X'].tolist()
    pY = c_df['Y'].tolist()
    points = np.stack([pX, pY]).T
    model = KMeans(n_clusters=32, random_state=42)
    print("Clustering...")
    model.fit(points)
    print("generating labels...")
    cluster_labels = model.predict(points)
    return cluster_labels

def compute_edge_cost(coord_a, coord_b):
    return np.sqrt(np.sum(np.square((coord_a - coord_b))))

# we need some way to "score" a path, which will be an array of integer
# city IDs, and "cities" is a dict of ID to numpy array of coords.
# Right now we're not including the prime penalty yet, though we'll add that later.
def compute_path_cost(input_path, cities, debug=False):
    total = 0.0
    cur = np.array([0.0, 0.0])
    i = 0
    for city_id in input_path:
        city_coord = cities[city_id]
        edge_cost = compute_edge_cost(city_coord, cur)
        total += edge_cost
        if debug:
            print("CUR: ", cur, "NEXT: ", city_coord, "COST:", edge_cost)
        cur = city_coord
        i += 1
        if debug and i > 20:
            break
    return total
