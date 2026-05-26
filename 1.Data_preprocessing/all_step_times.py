import os
import numpy as np

def extract_step_times(folder_path):
    """
    Extract time values for Step1_S, Step2_S, Step3_S, remove the first one, return three lists
    """
    step1_times = []
    step2_times = []
    step3_times = []

    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        print(f"Warning: '{folder_path}' path does not exist or is not a folder.")
        return step1_times, step2_times, step3_times

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path) and filename.lower().endswith('.txt'):
            if filename.startswith('Step-1_S'):
                time_value = extract_time_from_filename(filename)
                if time_value is not None:
                    step1_times.append(time_value)
            elif filename.startswith('Step-2_S'):
                time_value = extract_time_from_filename(filename)
                if time_value is not None:
                    step2_times.append(time_value)
            elif filename.startswith('Step-3_S'):
                time_value = extract_time_from_filename(filename)
                if time_value is not None:
                    step3_times.append(time_value)

    # Sort
    step1_times.sort()
    step2_times.sort()
    step3_times.sort()

    # Remove the first one
    step1_times = step1_times[1:] if len(step1_times) > 1 else []
    step2_times = step2_times[1:] if len(step2_times) > 1 else []
    step3_times = step3_times[1:] if len(step3_times) > 1 else []

    return step1_times, step2_times, step3_times

def extract_time_from_filename(filename):
    """
    Extract time value from filename
    """
    try:
        name = os.path.splitext(filename)[0]
        parts = name.split('_')
        time_str = parts[-1]
        return float(time_str)
    except (IndexError, ValueError):
        print(f"Warning: Cannot extract time value from '{filename}'")
        return None

def filter_step3_to_match_step1(step3_times, step1_times, target_count=6):
    """
    Select 6 from step3_times such that (t3 - 2.0) is closest to step1_times
    """
    if len(step1_times) == 0:
        return []

    eval_values = [(t3, t3 - 2.0) for t3 in step3_times]
    scored = []
    for t3, eval_val in eval_values:
        min_distance = min(abs(eval_val - t1) for t1 in step1_times)
        scored.append((t3, min_distance))
    
    scored.sort(key=lambda x: x[1])
    selected_t3 = [item[0] for item in scored[:target_count]]
    return sorted(selected_t3)

# === Main program: iterate from 0001 to 0100 ===
base_folder = r"F:\F410S_100"

result_dict = {}

# Generate folder names from 0001 to 0100
for i in range(1, 101):
    folder_name = f"{i:04d}"  # Format as 0001, 0002, ..., 0100
    folder_path = os.path.join(base_folder, folder_name)

    print(f"Processing: {folder_name}")

    # Extract time points
    step1, step2, step3 = extract_step_times(folder_path)

    # Filter Step-3_S to 6 best matches with Step-1_S
    if len(step1) > 0 and len(step3) >= 6:
        step3_filtered = filter_step3_to_match_step1(step3, step1, target_count=6)
    elif len(step3) == 0:
        step3_filtered = []
    else:
        # Less than 6, but has data, take the first 6 (or all)
        step3_filtered = sorted(step3)[:6]

    # Store in dictionary
    result_dict[folder_name] = {
        "Step-1_T": np.array(step1),
        "Step-2_T": np.array(step2),
        "Step-3_T": np.array(step3_filtered)  # Use filtered result
    }

output_folder = r"E:\Project\SGMW\model\F410S_18_100\Data"

# === Save results as .npz file ===
output_path = os.path.join(output_folder, "all_step_times.npz")

# Convert dict to key-value pairs suitable for np.savez
arrays_dict = {k: v for k, v in result_dict.items()}  # Directly use dict since each value is already np.array

# Save as .npz file
np.savez_compressed(output_path, **arrays_dict)  # Use savez_compressed for compression effect

print(f"\n✅ All data extracted and saved to:\n{output_path}")


