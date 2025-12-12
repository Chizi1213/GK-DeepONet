import numpy as np
import os

# Path settings
Data_path = '/home/zhangchi/data/SGMW/F410S_100/Data'
Data_morph_path = '/home/zhangchi/data/SGMW/F410S_100/Data/Morph'
coord_u_file = os.path.join(Data_path, 'Coord_U.npz')
displacement_file = os.path.join(Data_path, 'displacement.npz')
times_file = os.path.join(Data_path, 'step_times.npz')

# ===== 1. Load coordinate data (Coord_U.npz) =====
print("Loading Coord_U.npz...")
with np.load(coord_u_file, allow_pickle=True) as data:
    Coord_U = {key: data[key] for key in data.keys()}  # keys: '0' to '99'

print(f"Loaded coordinate data for {len(Coord_U)} experiments")

# Infer number of nodes from first experiment
first_coord = Coord_U['0']
if first_coord.shape[1] == 4:
    coords_3d = first_coord[:, 1:4].astype(np.float32)
else:
    coords_3d = first_coord.astype(np.float32)
n_nodes = coords_3d.shape[0]
print(f"Detected number of nodes: {n_nodes}")

# ===== 2. Load displacement data =====
print("🔍 Loading all_displacement.npz...")
with np.load(displacement_file, allow_pickle=True) as data:
    all_data_dict = {key: data[key] for key in data.keys()}
print(f"Loaded {len(all_data_dict)} displacement entries")

# ===== 3. Load time data =====
print("🔍 Loading all_step_times.npz...")
with np.load(times_file, allow_pickle=True) as data:
    time_data = {key: data[key].item() for key in data.keys()}  # keys: '0001' to '0100'
print(f"Loaded time info for {len(time_data)} experiments")

# Use all 100 experiments (indices 0 to 99)
indices = np.arange(100)  # Equivalent to np.linspace(0, 99, 100, dtype=int)
print(f"Processing all 100 experiments (indices 0–99)")

# Storage
output_pos = []   # (18, n_nodes, 3) per experiment
output_dis = []   # (18, n_nodes, 1) per experiment

# Process each experiment
for idx in indices:
    num_exp_int = idx + 1  # 1-based experiment number
    num_exp = f"{num_exp_int:04d}"
    coord_key = str(idx)   # Coord_U uses string keys '0', '1', ..., '99'

    print(f"Processing experiment: {num_exp}")

    # Load coordinates
    if coord_key not in Coord_U:
        raise KeyError(f"Coordinate data missing for experiment {num_exp} (key={coord_key})")
    
    coord_matrix = Coord_U[coord_key]
    if coord_matrix.shape[1] == 4:
        coords = coord_matrix[:, 1:4].astype(np.float32)
    else:
        coords = coord_matrix.astype(np.float32)

    if coords.shape[0] != n_nodes:
        raise ValueError(f"Node count mismatch in experiment {num_exp}")

    pos_time = np.tile(coords, (18, 1, 1))  # (18, n_nodes, 3)
    output_pos.append(pos_time)

    # Load time points
    if num_exp not in time_data:
        raise KeyError(f"Time data missing for experiment {num_exp}")
    
    data_dict = time_data[num_exp]
    steps = {
        'Step-1': data_dict.get('Step-1_T', []),
        'Step-2': data_dict.get('Step-2_T', []),
        'Step-3': data_dict.get('Step-3_T', [])
    }

    dis_time_list = []
    for step_key, time_array in steps.items():
        if len(time_array) < 6:
            raise ValueError(f"Experiment {num_exp} {step_key} has fewer than 6 time points")

        t_list = time_array[:6]
        for t in t_list:
            dict_key = f"{num_exp}_{step_key}_U_{t:.4f}"
            if dict_key not in all_data_dict:
                raise KeyError(f"Displacement data missing: {dict_key}")

            disp_mat = all_data_dict[dict_key]
            if disp_mat.shape[1] == 4:
                ux = disp_mat[:, 1].astype(np.float32)
                uy = disp_mat[:, 2].astype(np.float32)
                uz = disp_mat[:, 3].astype(np.float32)
            else:
                ux = disp_mat[:, 0].astype(np.float32)
                uy = disp_mat[:, 1].astype(np.float32)
                uz = disp_mat[:, 2].astype(np.float32)

            dis_mag = np.sqrt(ux**2 + uy**2 + uz**2)[:, np.newaxis]  # (n_nodes, 1)
            dis_time_list.append(dis_mag)

    assert len(dis_time_list) == 18, f"Expected 18 frames, got {len(dis_time_list)} in {num_exp}"
    dis_time = np.stack(dis_time_list, axis=0)  # (18, n_nodes, 1)
    output_dis.append(dis_time)

# Concatenate all experiments
final_pos = np.concatenate(output_pos, axis=0)  # (1800, n_nodes, 3)
final_dis = np.concatenate(output_dis, axis=0)  # (1800, n_nodes, 1)

print(f"final_pos.shape: {final_pos.shape}")
print(f"final_dis.shape: {final_dis.shape}")

# Save main data
save_path = os.path.join(Data_morph_path, 'Time_Coord_Dis.npz')
np.savez_compressed(save_path, pos=final_pos, dis=final_dis)
print(f"Saved to: {save_path}")
