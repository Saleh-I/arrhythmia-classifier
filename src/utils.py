import os
import wfdb
import numpy as np
from tqdm import tqdm
from collections import Counter

def pre_process(data_dir):
    '''

    Args:
        data_dir:

    Returns: X (numpy array) (num of samples , 256, 12)
             y  (numpy array)   (num of samples,)

    '''
    fs = 257  # Sampling frequency.
    window_size = fs  # 1 second
    X_all = []
    y_all = []

    # List all .hea files
    hea_files = [f for f in os.listdir(data_dir) if f.endswith('.hea')]

    # Process each record
    for hea_file in tqdm(hea_files, desc="Processing records"):
        record_name = os.path.splitext(hea_file)[0] # I01, I02, ..I0n
        record_path = os.path.join(data_dir, record_name) # ../data/files\I01

        try:
            record = wfdb.rdrecord(record_path)
            annotation = wfdb.rdann(record_path, 'atr')
        except Exception as e:
            print(f"Skipping {record_name} due to error: {e}")
            continue

        signal = record.p_signal
        labels = annotation.symbol
        positions = annotation.sample

        for idx, pos in enumerate(positions):
            start = pos - window_size // 2
            end = pos + window_size // 2

            # Skip if near edges
            if start < 0 or end > len(signal):
                continue

            # Extract window
            window = signal[start:end, :]
            X_all.append(window)
            y_all.append(labels[idx])

    # Convert to arrays
    X_all = np.array(X_all)
    y_all = np.array(y_all)
    # Normalize (z-score) each sample per lead
    X_all = (X_all - X_all.mean(axis=1, keepdims=True)) / (X_all.std(axis=1, keepdims=True) + 1e-8)

    # Remove non-beat markers like '+'
    valid_mask = y_all != '+'
    X_clean = X_all[valid_mask]
    y_clean = y_all[valid_mask]

    return X_clean, y_clean




# def calculate_class_weights(y):
#
#     label_counts = Counter(y)
#
#     # Map each label to the corresponding class
#     label_to_class = {
#         'N': 0, 'R': 0, 'L': 0, 'n': 0, 'B': 0,  # Normal
#         'A': 1, 'S': 1, 'j': 1,  # SVEB
#         'V': 2,  # VEB
#         '+': 3, 'F': 3, 'Q': 3  # Other
#     }
#
#     # Convert to class indices
#     class_counts = np.zeros(4)
#     for label in y:
#         class_idx = label_to_class.get(label, 3)
#         class_counts[class_idx] += 1
#
#     # Compute class weights
#     class_weights = 1.0 / (class_counts + 1e-5)
#     class_weights = class_weights / class_weights.sum()
#
#     return class_weights