#!/usr/bin/env python3
"""
Create a minimal ECG dataset pickle file for testing
"""
import pickle
import numpy as np

# Create minimal ECG data with the right structure
# Based on data_loaders.py line 442-459, we need:
# - 'data': list of ECG signals (1000-length time series)
# - 'label': list of labels ('N', 'A', 'O', '~')

np.random.seed(42)

# Create 100 synthetic ECG traces (enough for batch_size=4 training)
# Each trace is 3000 samples long (will be windowed by dataloader)
data = []
labels = []
label_types = ['N', 'A', 'O', '~']

for i in range(100):
    # Synthetic ECG: random walk with some periodicity
    ecg_signal = np.cumsum(np.random.randn(3000)) * 0.1
    ecg_signal += np.sin(np.linspace(0, 30*np.pi, 3000)) * 0.5  # Add periodic component
    data.append(ecg_signal.astype(np.float32))
    labels.append(label_types[i % 4])

# Create the expected pickle structure
# Note: data must be a numpy array, not a list (slide_and_cut expects .shape attribute)
ecg_data = {
    'data': np.array(data),  # Convert list to numpy array
    'label': labels
}

# Save to the expected location
output_path = 'datasets/challenge2017.pkl'
with open(output_path, 'wb') as f:
    pickle.dump(ecg_data, f)

print(f"✅ Created minimal ECG dataset at {output_path}")
print(f"   - {len(data)} ECG traces")
print(f"   - Signal length: {len(data[0])} samples")
print(f"   - Labels: {set(labels)}")
print(f"   - File size: {len(pickle.dumps(ecg_data))/1024:.1f} KB")
print()
print("This is a synthetic dataset for testing LoRA integration only.")
print("For real experiments, download the full PhysioNet Challenge 2017 dataset.")
