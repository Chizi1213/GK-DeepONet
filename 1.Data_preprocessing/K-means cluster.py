import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

# -------------------------------
# Normalization and translation functions
# -------------------------------
def normalize_and_translate(points):
    """
    Normalize points to approximately [-0.5, 0.5] × [-0.5, 0.5] × [0, 1]
    """
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    bbox_size = max_coords - min_coords
    scale_factor = 1.0 / np.max(bbox_size)

    normalized = (points - min_coords) * scale_factor

    # Align bottom to Z=0 (grounding)
    min_z = np.min(normalized[:, 2])
    normalized[:, 2] -= min_z

    # Shift bottom center to origin
    bottom_mask = normalized[:, 2] <= (min_z + 1e-6)
    bottom_center_x = np.mean(normalized[bottom_mask, 0])
    bottom_center_y = np.mean(normalized[bottom_mask, 1])

    normalized[:, 0] -= bottom_center_x
    normalized[:, 1] -= bottom_center_y

    return normalized, min_coords, scale_factor


# -------------------------------
# Path and parameter settings
# -------------------------------
Data_path = '/home/zhangchi/data/SGMW/F410S_100/Data/'
Data_morph_path = Data_path + 'Morph/'
Data_ratio_path = Data_path + 'Ratio/'

Resampledata_path = Data_morph_path + f'Time_Coord_Dis.npz'

# Load data
tmp = np.load(Resampledata_path)
Coords_flt = tmp['pos']  # shape: (N_sample, N_node, 3)

# Get coordinates of the first sample
coords_first = Coords_flt[0]  # (N_node, 3)
n_nodes = coords_first.shape[0]
print(f"Original number of nodes: {n_nodes}")

# Normalize
normalized_first, min_coords, scale_factor = normalize_and_translate(coords_first)
print(f"Coordinate range after normalization:")
print(f"  X: [{normalized_first[:,0].min():.3f}, {normalized_first[:,0].max():.3f}]")
print(f"  Y: [{normalized_first[:,1].min():.3f}, {normalized_first[:,1].max():.3f}]")
print(f"  Z: [{normalized_first[:,2].min():.3f}, {normalized_first[:,2].max():.3f}]")


# -------------------------------
# Define sampling ratios and spatial scope
# -------------------------------
nodes_ratios = [0.05, 0.1, 0.2, 0.5]


################################# K-means clustering sampling ####################################
for ratio in nodes_ratios:
    print(f"\n--- Processing node sampling ratio = {ratio} ---")
    n_select = int(n_nodes * ratio)
    
    if n_select <= 0:
        raise ValueError(f"n_select is {n_select}; please check n_nodes and ratio.")

    # Perform KMeans clustering
    print(f"    Running KMeans clustering with n_clusters = {n_select}...")
    kmeans = KMeans(n_clusters=n_select, random_state=2025).fit(normalized_first)

    # Get cluster labels for each point
    labels = kmeans.labels_  # shape: (n_nodes,)

    # Get cluster centers
    cluster_centers = kmeans.cluster_centers_  # shape: (n_select, 3)

    # Initialize list to store selected indices
    idx_selected = []

    # For each cluster, select the original point closest to its center
    for i in range(n_select):
        # Find indices of points belonging to cluster i
        points_in_cluster = np.where(labels == i)[0]
        
        if len(points_in_cluster) == 0:
            # Should not happen in practice, but included for safety
            continue

        # Compute distances from these points to the cluster center
        points_coords = normalized_first[points_in_cluster]
        center = cluster_centers[i]
        distances = np.linalg.norm(points_coords - center, axis=1)

        # Select the closest point (in terms of original index in coords_first)
        closest_idx_in_cluster = points_in_cluster[np.argmin(distances)]
        idx_selected.append(closest_idx_in_cluster)

    # Convert to numpy array and sort
    idx_selected = np.array(idx_selected)
    idx_selected = np.sort(idx_selected)

    # Save index file
    index_file = Data_ratio_path + f'Dis_node_k_indices_ratio{str(ratio)}.npy'
    np.save(index_file, idx_selected)
    print(f"Saved node indices to: {index_file}, count: {len(idx_selected)}")