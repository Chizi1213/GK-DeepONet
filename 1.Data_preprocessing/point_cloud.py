import pandas as pd
import numpy as np
import os

num = 100  # Number of samples
data_path = "E:/Project/SGMW/model/F410S_18_100/Data/"
dataset_path = "F:/F410S_100/"  # Each num_exp should have a corresponding folder

# Assumed predefined variables
step_key = 'Step-1'
num_exp = '0001'

# Only process field 'U'
field_keys = ['U']
time_val = 0  # Assume the same time value for all fields

# Dictionary to store results
id_dict = {}

# Loop over each field key (only 'U' in this case)
for field_key in field_keys:
    filename = f"{step_key}_{field_key}_{time_val:.4f}.txt"
    file_path = os.path.join(dataset_path, num_exp, filename)

    print(f"Processing file: {file_path}")

    # Initialize list to store node IDs for this field
    id_list = []

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    # Read the file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        continue

    # Skip the first line (header)
    for line in lines[1:]:
        row = line.strip().split(',')
        if len(row) < 2:
            continue  # Skip empty or malformed lines

        try:
            node_id = int(row[0].strip())
            # For 'U', just collect the node ID
            id_list.append([node_id])
        except ValueError as e:
            print(f"Parse error in {file_path}: {line}, error: {e}")
            continue

    # Convert to numpy array and store in dictionary
    id_dict[f'id_{field_key}'] = np.array(id_list)

# Extract id_U (as a fallback, use empty array if not present)
id_U = id_dict.get('id_U', np.array([]))

# Optional: print shape
print("id_U shape:", id_U.shape)

# Extract 1D set of node IDs for fast lookup
if id_U.size > 0:
    node_ids_U = set(id_U.flatten())
else:
    node_ids_U = set()

# Dictionary to store filtered coordinates for 'U'
Coord_U = {}

# Iterate through each experiment folder
for i in range(1, num + 1):
    num_exp = str(i).zfill(4)
    coord_path = os.path.join(dataset_path, num_exp)
    para_name = 'nodes.txt'
    file_path = os.path.join(coord_path, para_name)

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()[1:]  # Skip header

        coordinates = []  # Store valid [id, x, y, z]
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 4:
                continue
            try:
                node_id = float(parts[0])
                x, y, z = map(float, parts[1:4])
                coordinates.append([node_id, x, y, z])
            except (ValueError, IndexError):
                continue

        if not coordinates:
            print(f"Warning: No valid coordinate data in {file_path}")
            continue

        value_matrix = np.array(coordinates)  # Shape: (N, 4), columns: [id, x, y, z]
        node_ids_in_file = value_matrix[:, 0]  # All node IDs in this file

        # Filter nodes that are in node_ids_U
        mask_U = np.isin(node_ids_in_file, list(node_ids_U))
        selected_U = value_matrix[mask_U] if mask_U.any() else np.zeros((0, 4))

        # Store using zero-based key
        key = str(i - 1)
        Coord_U[key] = selected_U

        print(f"Processed {file_path}: Found {len(selected_U)} U-nodes")

    except FileNotFoundError:
        print(f"Warning: File not found: {file_path}")
        continue
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        continue

# Save to .npz file
Coord_U_path = os.path.join(data_path, 'Coord_U.npz')
np.savez(Coord_U_path, **Coord_U)
print(f"Coord_U saved to: {Coord_U_path}")