import os
import numpy as np

# Path settings
Data_path = "E:/Project/SGMW/model/F410S_18_100/Data/"
Dataset_path = "F:/F410S_100/"  # Each num_exp should have a corresponding subfolder

# Load time data
times_path = os.path.join(Data_path, 'all_step_times.npz')
loaded_data = np.load(times_path, allow_pickle=True)

# Dictionary to store all displacement data
all_data = {}

# Loop over experiments: 0001 to 0100
for num_exp_int in range(1, 101):
    num_exp = f"{num_exp_int:04d}"  # Format as 0001, 0002, ..., 0100

    print(f"Processing experiment {num_exp}...")

    # Check if this experiment exists in loaded time data
    if num_exp not in loaded_data:
        print(f"Warning: {num_exp} not found in time data, skipping...")
        continue

    data_dict = loaded_data[num_exp].item()  # Convert to standard dict

    # Define steps and their time arrays
    steps = {
        'Step-1': data_dict.get('Step-1_T', []),
        'Step-2': data_dict.get('Step-2_T', []),
        'Step-3': data_dict.get('Step-3_T', [])
    }

    # Process each step
    for step_key, time_array in steps.items():
        if len(time_array) < 6:
            print(f"Warning: {num_exp} {step_key} has fewer than 6 time points, skipping step...")
            continue

        # Take the first 6 time points
        t_list = time_array[:6]  # t1 to t6

        # Only process displacement field ('U')
        field_key = 'U'
        for i, time_val in enumerate(t_list, start=1):
            # Construct filename: e.g., Step-1_U_0.1000.txt
            filename = f"{step_key}_{field_key}_{time_val:.4f}.txt"
            file_path = os.path.join(Dataset_path, num_exp, filename)

            # Check if file exists
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}, skipping...")
                continue

            # Read and parse the file
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                data_matrix = []
                # Skip header line
                for line in lines[1:]:
                    row = line.strip().split(',')
                    if len(row) < 4:
                        continue  # Skip malformed or empty lines

                    try:
                        node_id = int(row[0].strip())
                        u1 = float(row[1].strip())
                        u2 = float(row[2].strip())
                        u3 = float(row[3].strip())
                        data_matrix.append([node_id, u1, u2, u3])
                    except ValueError as e:
                        print(f"Parse error in {file_path}: {line}, error: {e}")
                        continue

                # Convert to numpy array
                data_matrix = np.array(data_matrix)

                # Create dictionary key: e.g., "0001_Step-1_U_0.1000"
                dict_key = f"{num_exp}_{step_key}_{field_key}_{time_val:.4f}"

                # Store in global dictionary
                all_data[dict_key] = data_matrix

            except Exception as e:
                print(f"Error reading {file_path}: {e}")

# Save all displacement data to .npz file
output_path = os.path.join(Data_path, "displacement.npz")
np.savez(output_path, **all_data)
print(f"All displacement data saved to {output_path}")