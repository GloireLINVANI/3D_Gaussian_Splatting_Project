from plyfile import PlyData
import numpy as np
from scipy.spatial import KDTree
import random

COLORS = [[252,199,55],[242,107,15],[231,56,121],[126,24,145], [247,44,91],[255,116,139],[167,212,119],[228,241,172]]

def get_vertex_info(plydata):
    """
    plyData: obtained from .ply file
    returns an np array of position
    """
    vertices = plydata['vertex']
    #positions
    x = vertices['x']
    y = vertices['y']
    z = vertices['z']

    #colors
    c1 = vertices['f_dc_0']
    c2 = vertices['f_dc_1']
    c3 = vertices['f_dc_2']

    points = np.column_stack((x, y, z)) 
    colors = np.column_stack((c1, c2, c3)) 

    print(points)
    print(colors)
    return points, colors


def generate3d_tree(points):
    kd_tree = KDTree(points)

    query_point = [0.5396436, 0.7212234, 0.49657318]

    # Find the 2 nearest neighbors
    distances, indices = kd_tree.query(query_point, k=4)

    print("Distances to neighbors:", distances)
    print("Indices of neighbors:", indices)


def k_means_kd_tree(data, k, colors, max_iter=100, tol=1e-4):
    """
    Perform k-means clustering using k-d tree for nearest centroid search.
    
    Parameters:
        data (ndarray): Input data of shape (N, D), where N is the number of points and D is the dimensionality.
        k (int): Number of clusters.
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance for centroid change to consider convergence.
    
    Returns:
        centroids (ndarray): Final centroids of shape (k, D).
        labels (ndarray): Cluster labels for each point of shape (N,).

    """
    # Step 1: Initialize centroids randomly
    N, D = data.shape
    centroids = data[np.random.choice(N, k, replace=False)]

    for iteration in range(max_iter):
        print(iteration)
        # Step 2: Build a k-d tree for centroids
        kd_tree = KDTree(centroids)

        # Step 3: Assign points to nearest centroid
        labels = np.zeros(N, dtype=int)
        for i, point in enumerate(data):
            _, nearest_idx = kd_tree.query(point)
            labels[i] = nearest_idx

        # Step 4: Update centroids
        new_centroids = np.array([
            data[labels == cluster].mean(axis=0) if np.any(labels == cluster) else centroids[cluster]
            for cluster in range(k)
        ])

        # Check for convergence
        print(np.linalg.norm(new_centroids - centroids))
        if np.linalg.norm(new_centroids - centroids) < tol:
            print(f"Converged after {iteration + 1} iterations.")
            break

        centroids = new_centroids

    #change the color of the gaussians for each cluster
    #Assign points to nearest centroid
    labels = np.zeros(N, dtype=int)
    kd_tree = KDTree(centroids)
    for i, point in enumerate(data):
        _, nearest_idx = kd_tree.query(point)
        labels[i] = nearest_idx

    #color by cluster
    for cluster in range(k):
        cluster_color = np.array(COLORS[cluster%len(COLORS)])
        colors[labels == cluster] = cluster_color

    return centroids, labels, colors



def k_means_with_color(points, k, colors, max_iter=100, tol=1e-4):
    # Step 1: Initialize centroids randomly
    data = np.concatenate((points, colors), axis=1)
    N, D = data.shape
    centroids = data[np.random.choice(N, k, replace=False)]

    for iteration in range(max_iter):
        print(iteration)
        # Step 2: Build a k-d tree for centroids
        kd_tree = KDTree(centroids)

        # Step 3: Assign points to nearest centroid
        labels = np.zeros(N, dtype=int)
        for i, point in enumerate(data):
            _, nearest_idx = kd_tree.query(point)
            labels[i] = nearest_idx

        # Step 4: Update centroids
        new_centroids = np.array([
            data[labels == cluster].mean(axis=0) if np.any(labels == cluster) else centroids[cluster]
            for cluster in range(k)
        ])

        # Check for convergence
        print(np.linalg.norm(new_centroids - centroids))
        if np.linalg.norm(new_centroids - centroids) < tol:
            print(f"Converged after {iteration + 1} iterations.")
            break

        centroids = new_centroids

    #change the color of the gaussians for each cluster
    #Assign points to nearest centroid
    labels = np.zeros(N, dtype=int)
    kd_tree = KDTree(centroids)
    for i, point in enumerate(data):
        _, nearest_idx = kd_tree.query(point)
        labels[i] = nearest_idx

    #color by cluster
    for cluster in range(k):
        cluster_color = np.array([random.random(), random.random(), random.random()])
        colors[labels == cluster] = cluster_color

    return centroids, labels, colors



def set_color(ply_data, colors, modified_path):
    """
    change the difuse color of the ply_data
    """
    vertices = plydata['vertex']
    vertices['f_dc_0'] = colors[:, 0]
    vertices['f_dc_1'] = colors[:, 1]
    vertices['f_dc_2'] = colors[:, 2]

    with open(modified_path, 'wb') as f:
        print("writing new data")
        PlyData(plydata.elements).write(f)



if __name__ == "__main__":
    file_path = r"C:\Users\onurb\master\computer_vision\projet\CSC_51073_EP-Project\CSC_51073_EP-Project\data\point_cloud.ply"
    modified_path = r'3D_clustering\kmeans_color.ply'

    with open(file_path, 'rb') as f:
        plydata = PlyData.read(f)    

    points, colors = get_vertex_info(plydata)

    _,_, colors = k_means_with_color(points, 10, colors, max_iter=10)
    
    set_color(plydata, colors, modified_path)

