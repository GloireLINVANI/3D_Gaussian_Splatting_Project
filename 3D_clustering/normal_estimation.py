from plyfile import PlyData
import numpy as np
from scipy.spatial import KDTree
import random
from scipy.linalg import eigh
import math
import sys
from collections import deque


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

    return points, colors

def get_pos(ply_data):
    vertices = plydata['vertex']
    #positions
    x = vertices['x']
    y = vertices['y']
    z = vertices['z']

    points = np.column_stack((x, y, z))
    return points

def generate_sphere_ply(radius=1.0, subdivisions=50, filename="sphere.ply"):
    # Create vertices
    vertices = []
    for i in range(subdivisions + 1):
        theta = i * math.pi / subdivisions  # Latitude angle (0 to pi)
        for j in range(subdivisions):
            phi = j * 2.0 * math.pi / subdivisions  # Longitude angle (0 to 2pi)

            x = radius * math.sin(theta) * math.cos(phi)
            y = radius * math.sin(theta) * math.sin(phi)
            z = radius * math.cos(theta)

            vertices.append([x, y, z, 255, 0, 0])  # Append coordinates and color (red)

    vertices = np.array(vertices)

    # Write PLY file
    with open(filename, "w") as ply_file:
        # Header
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write(f"element vertex {len(vertices)}\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
        ply_file.write("end_header\n")

        # Vertices with colors
        for vertex in vertices:
            ply_file.write(f"{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f} {int(vertex[3])} {int(vertex[4])} {int(vertex[5])}\n")

    print(f"Sphere saved to {filename}")

def compute_normals(V1, k):
    """
    Compute normals using PCA on k-nearest neighbors.

    Parameters:
        V1 (ndarray): The input point cloud of shape (N, 3), where N is the number of points.
        k (int): Number of nearest neighbors to consider.

    Returns:
        normals (ndarray): Computed normals of shape (N, 3).
    """
    print("Calculating normals...")
    kd_tree = KDTree(V1)
    normals = np.zeros((V1.shape[0], 3))

    for i in range(V1.shape[0]):
        if i % 1000 == 0:
            print(f"Processing point {i}/{V1.shape[0]}")

        # Find k-nearest neighbors
        _, indices = kd_tree.query(V1[i], k)
        neighbors = V1[indices]
        #neighbors += np.random.normal(0, 0.1, neighbors.shape)

        # Compute centroid
        centroid = neighbors.mean(axis=0)

        # Center neighbors
        centered = neighbors - centroid

        # Compute covariance matrix
        covariance = np.dot(centered.T, centered)

        # Eigenvalue decomposition
        eigenvalues, eigenvectors = eigh(covariance)

        # Smallest eigenvector as normal
        normal = eigenvectors[:, 0]

        # Orient normal consistently
        pos = V1[i]
        #print(normal/np.linalg.norm(normal))
        if np.dot(normal, pos - centroid) > 0:
            normal = -normal  # Flip direction
        
        normals[i] = normal #*0.5+0.5 for visualisation

    # Normalize all normals
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    #print(np.unique(normals))
    print("Normal calculation complete.")
    return normals


def compute_residuals(V1, normals, k):
    """
    Compute residuals for a point cloud.

    Parameters:
        V1 (ndarray): Input point cloud of shape (N, 3).
        normals (ndarray): Computed normals of shape (N, 3).
        k (int): Number of nearest neighbors.

    Returns:
        residuals (ndarray): Residuals for each point of shape (N,).
    """
    print("Calculating residuals...")

    kd_tree = KDTree(V1)
    residuals = np.zeros(V1.shape[0])

    for i in range(V1.shape[0]):
        if i % 1000 == 0:
            print(f"Processing point {i}/{V1.shape[0]}")

        # Find k-nearest neighbors
        _, indices = kd_tree.query(V1[i], k)
        neighbors = V1[indices]

        # Compute centroid
        centroid = neighbors.mean(axis=0)

        # Compute residual as the orthogonal distance to the best-fit plane
        residuals[i] = abs(np.dot(normals[i], V1[i] - centroid))
    
    return residuals


def segmentation_3D(points, normals, residuals, residual_threshold, angle_threshold, k):
    """
    Segments a given point cloud using smoothness constraint.
    
    Arguments:
    points -- Numpy array of shape (N, 3) representing the point cloud.
    normals -- Numpy array of shape (N, 3) representing the normal vectors.
    residuals -- Numpy array of shape (N,) representing the residuals for each point.
    residual_threshold -- float, the threshold for the residual.
    angle_threshold -- float, the threshold for the smoothness constraint (in radians).
    
    Returns:
    segmented_regions -- List of segmented regions, where each region is a list of points.
    """
    # Initialize
    N = len(points)
    A = set(range(N))  # Available points list
    R = []  # Global region list
    
    # Build KDTree for fast nearest neighbor search
    kd_tree = KDTree(points)
    
    while A:
        # Initialize current region and seed list
        Rc = set()
        Sc = set()
        
        # Step 3: Select point with minimum residual from available points
        Pmin_idx = min(A, key=lambda idx: residuals[idx])
        Sc.add(Pmin_idx)
        Rc.add(Pmin_idx)
        A.remove(Pmin_idx)
        
        queue = deque(Sc)
        # Step 4: Region growing process
        #for seed_idx in list(Sc):
        while queue:
            seed_idx = queue.popleft()
            # Find nearest neighbors of current seed
            neighbors_idx = kd_tree.query(points[seed_idx], k)[1]
            for neighbor_idx in neighbors_idx:
                if neighbor_idx in A:
                    # Compute the smoothness condition (angle between normals)
                    cos_angle = np.abs(np.dot(normals[seed_idx], normals[neighbor_idx]))
                    #print(cos_angle)
                    if cos_angle > np.cos(angle_threshold):
                        Rc.add(neighbor_idx)
                        A.remove(neighbor_idx)
                        
                        # Check the residual condition
                        if residuals[neighbor_idx] < residual_threshold:
                            #Sc.add(neighbor_idx)
                            queue.append(neighbor_idx)

        # Step 6: Add current region to global segment list
        R.append(list(Rc))
    
    # Step 7: Sort regions by size
    R.sort(key=len, reverse=True)
    
    return R


def set_clusters(plydata, R, modified_path):
    """
    for each region set a color in a ply file
    R: region list
    modified_path: the path where the 
    """
    vertices = plydata['vertex']

    for Rc in R:
        r = np.array(Rc)
        vertices['f_dc_0'][r] = random.random()
        vertices['f_dc_1'][r] = random.random()
        vertices['f_dc_2'][r] = random.random()

    with open(modified_path, 'wb') as f:
        print("writing new data")
        PlyData(plydata.elements).write(f)


def set_normal(ply_data, normals, modified_path):
    """
    change the difuse color of the ply_data
    """
    vertices = plydata['vertex']
    vertices['f_dc_0'] = normals[:, 0]
    vertices['f_dc_1'] = normals[:, 1]
    vertices['f_dc_2'] = normals[:, 2]

    print(normals[:, 0] * 255)

    with open(modified_path, 'wb') as f:
        print("writing new data")
        PlyData(plydata.elements).write(f)

if __name__ == "__main__":
    file_path = r"data\point_cloud.ply"
    modified_path = r'3D_clustering\clustering.ply'

    with open(file_path, 'rb') as f:
        plydata = PlyData.read(f)    

    points = get_pos(plydata)

    normals = compute_normals(points, 2000)

    residuals = compute_residuals(points, normals, 2000)

    print(residuals)

    R = segmentation_3D(points, normals, residuals, residual_threshold=0.1, angle_threshold=0.05, k=10)

    print(f"number of segments: {len(R)}")

    set_clusters(plydata, R, modified_path)

    #set_normal(plydata, normals, modified_path)

    #generate_sphere_ply(subdivisions=80, filename=file_path)