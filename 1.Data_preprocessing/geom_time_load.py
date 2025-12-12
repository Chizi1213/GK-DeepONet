import pandas as pd
import numpy as np
import re

data_path = 'E:/Project/SGMW/model/F410S_18_100/Data/'
file_path = 'F:/F410S_100/odb_mapping.txt'

result_list = []

with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        line = line.strip()
        if not line:
            continue  # Skip empty lines
        parts = line.split(',')  # Split by comma
        if len(parts) < 2:
            continue

        first_col = parts[0]
        second_col = parts[1]

        if 'run_' in first_col:
            try:
                number_part = first_col.split('run_')[1].split('.odb')[0]
                number_part = number_part[-4:]  # Take last 4 digits
                result_list.append([int(number_part)])
            except (IndexError, ValueError):
                continue  # Skip malformed lines

# Convert to NumPy array (100 rows x 1 column)
mapping_index = np.array(result_list, dtype=object)

# Read DOE.xlsx file
doe_file_path = 'F:/F410S_18_490/DOE.xlsx'
doe_df = pd.read_excel(doe_file_path, index_col=0)

# Extract values from DOE corresponding to indices in mapping_index
doe_values = []
for idx in mapping_index[:, 0]:
    if idx in doe_df.index:
        doe_values.append(doe_df.loc[idx].values)
    else:
        doe_values.append(np.nan)  # Use NaN if index not found

# Convert extracted values to NumPy array
doe_matrix = np.array(doe_values, dtype=object)

# Combine experiment ID (first column) with DOE parameters
Geom_matrix = np.hstack((mapping_index[:, 0].reshape(-1, 1), doe_matrix))  # (ID [1-based], DOE params)


def CF_data(name_case: str):
    CF_path = f'F:/F410S_100/{name_case}/CFs.txt'
    
    frame_values = []
    cf_values = []
    
    prev_step = None
    last_time = None
    first_row = True  # Used to skip the very first point (Step-1, 0.0000)

    try:
        with open(CF_path, 'r') as file:
            lines = file.readlines()
        
        # Skip header line (assume first line is header)
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            
            # Parse line using regex: e.g., "Step-1, 0.1000, 5.000000"
            match = re.match(r"Step-(\d+)\s*,\s*([^,]+)\s*,\s*([^,]+)", line)
            if not match:
                print(f"Warning: Failed to parse line - {line}")
                continue
            
            step = int(match.group(1))
            try:
                time_val = float(match.group(2))
                value = float(match.group(3))
            except ValueError:
                print(f"Warning: Failed to convert numeric values - {line}")
                continue

            # --- Data cleaning logic ---

            # 1. Skip the very first data point (Step-1, time ≈ 0)
            if first_row and step == 1 and abs(time_val) < 1e-5:
                first_row = False
                continue  # Skip this point
            first_row = False

            # 2. If entering a new Step and time equals previous Step's end time → skip duplicate start point
            if prev_step is not None and step > prev_step:
                if abs(time_val - last_time) < 1e-5:
                    continue  # Skip duplicate

            # Keep valid data
            frame_values.append(time_val)
            cf_values.append(value)

            # Update state
            prev_step = step
            last_time = time_val

    except FileNotFoundError:
        print(f"Error: File not found - {CF_path}")
        return np.array([]).reshape(0, 2)
    except Exception as e:
        print(f"Error reading or processing file: {e}")
        return np.array([]).reshape(0, 2)

    # ================== Build CF matrix ==================
    if len(frame_values) == 0:
        print("Warning: No valid data extracted.")
        return np.array([]).reshape(0, 2)

    frame_matrix = np.array(frame_values).reshape(-1, 1)
    cf_matrix = np.array(cf_values).reshape(-1, 1)
    CF = np.hstack((frame_matrix, cf_matrix))  # shape: (N, 2)

    return CF


############################### Load time data ############################
times_path = data_path + 'all_step_times.npz'
loaded_data = np.load(times_path, allow_pickle=True)

# Assume Geom_matrix is defined: shape (100, N), each row = [ID, param1, param2, ...]
experiment_ids = [f"{i:04d}" for i in range(1, 101)]  # ['0001', ..., '0100']

# List to collect all processed matrices
all_matrices = []

# Outer loop: iterate over each experiment
for num_exp in experiment_ids:
    print(f"Processing experiment: {num_exp}")

    try:
        data_dict = loaded_data[num_exp].item()  # Get time points for this experiment
    except KeyError:
        print(f"Experiment {num_exp} not found in all_step_times.npz, skipping.")
        continue

    # Define steps and their time arrays
    steps = {
        'Step-1': data_dict.get('Step-1_T', []),
        'Step-2': data_dict.get('Step-2_T', []),
        'Step-3': data_dict.get('Step-3_T', [])
    }

    # Get DOE parameters for current experiment
    target_id = int(num_exp) - 1
    try:
        target_doe = Geom_matrix[target_id][1:]  # Exclude first column (ID)
    except IndexError:
        print(f"DOE index {target_id} out of bounds, skipping {num_exp}")
        continue

    # Load CF data for this experiment
    cf_data = CF_data(num_exp)
    if cf_data.shape[0] == 0:
        print(f"CF data is empty for {num_exp}, skipping.")
        continue

    # Build fast lookup dict: time → CF value
    time_to_data = dict(zip(cf_data[:, 0], cf_data[:, 1]))

    # Inner loop: process each step
    for step_key, time_array in steps.items():
        if len(time_array) < 6:
            print(f"{num_exp} {step_key} has fewer than 6 time points, skipping.")
            continue

        t_list = np.array(time_array[:6])  # Take first 6 time points
        n_time = len(t_list)

        # Repeat DOE parameters for each time point
        doe_matrix = np.tile(target_doe, (n_time, 1))  # shape: (6, num_params)

        # Match CF values at each time point
        matched_data = []
        for t in t_list:
            if t in time_to_data:
                matched_data.append(time_to_data[t])
            else:
                print(f"Time point {t} not found in CF data (exp {num_exp}, {step_key}), using NaN")
                matched_data.append(np.nan)

        # Construct matrix: [DOE_params, time, CF_value]
        new_matrix = np.column_stack((doe_matrix, t_list, matched_data))  # shape: (6, num_params + 2)

        # Append to global list
        all_matrices.append(new_matrix)

# Stack all matrices vertically
if all_matrices:
    final_matrix = np.vstack(all_matrices)
    print(f"Processing complete! Final matrix shape: {final_matrix.shape}")
else:
    print("No valid matrices generated.")
    # Create an empty matrix with expected number of columns (adjust if needed)
    num_cols = len(Geom_matrix[0]) + 1  # Assuming Geom_matrix includes ID + params
    final_matrix = np.array([]).reshape(0, num_cols)

# Save result
output_path = data_path + 'geom_time_load.npy'
np.save(output_path, final_matrix)
print(f"Saved to: {output_path}")