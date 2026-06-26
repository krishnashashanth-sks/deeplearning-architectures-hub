import os
import requests
import zipfile
from utils import *
import pickle 
from tqdm.auto import tqdm
from torch.utils.data import Dataset,random_split
import torch

# Define paths and URLs (using previously defined variables if available, otherwise defining them)
base_dir = './data'
download_url = 'http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip' # Updated URL
zip_file_path = os.path.join(base_dir, 'ModelNet10.zip')
extraction_path = os.path.join(base_dir, 'ModelNet10')

# Create the base directory if it doesn't exist
os.makedirs(base_dir, exist_ok=True)

print(f"Downloading ModelNet10 from {download_url}...")

# Download the dataset
response = requests.get(download_url, stream=True)
if response.status_code == 200:
    with open(zip_file_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Download complete. File saved to {zip_file_path}")

    # Extract the dataset
    print(f"Extracting {zip_file_path} to {extraction_path}...")
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extraction_path)
    print("Extraction complete.")

    # Verify extraction by listing contents
    print("Contents of extracted directory:")
    for item in os.listdir(extraction_path):
        print(f"- {item}")

else:
    print(f"Failed to download ModelNet10. Status code: {response.status_code}")

modelnet_path = './data/ModelNet10/ModelNet10'

# Lists to store voxel grids and labels
all_voxel_grids = []
all_labels = []

# Mapping class names to integer labels
class_to_idx = {}
current_idx = 0

print(f"Processing files in {modelnet_path}...")

# Iterate through subdirectories (classes)
for class_name in os.listdir(modelnet_path):
    class_path = os.path.join(modelnet_path, class_name)

    if os.path.isdir(class_path):
        if class_name not in class_to_idx:
            class_to_idx[class_name] = current_idx
            current_idx += 1
        label = class_to_idx[class_name]

        # Iterate through train/test subdirectories
        for split_type in ['train', 'test']:
            split_path = os.path.join(class_path, split_type)
            if os.path.isdir(split_path):
                # Iterate through .off files
                for filename in os.listdir(split_path):
                    if filename.endswith('.off'):
                        off_filepath = os.path.join(split_path, filename)
                        voxel_grid = mesh_to_voxel_grid(off_filepath)
                        if voxel_grid is not None:
                            all_voxel_grids.append(voxel_grid)
                            all_labels.append(label)

print(f"Finished processing. Total voxel grids: {len(all_voxel_grids)}")
print(f"Total labels: {len(all_labels)}")
print(f"Detected classes: {class_to_idx}")

processed_data_path = './data/processed_modelnet10.pkl' # Path to save processed data

# Lists to store voxel grids and labels
all_voxel_grids = []
all_labels = []

# Mapping class names to integer labels
class_to_idx = {}
current_idx = 0

print(f"Processing files in {modelnet_path}...")

# Check if processed data already exists to resume or skip
if os.path.exists(processed_data_path):
    print(f"Loading previously processed data from {processed_data_path}...")
    with open(processed_data_path, 'rb') as f:
        loaded_data = pickle.load(f)
        all_voxel_grids = loaded_data['voxel_grids']
        all_labels = loaded_data['labels']
        class_to_idx = loaded_data['class_to_idx']
        current_idx = loaded_data['current_idx']
    print(f"Loaded {len(all_voxel_grids)} voxel grids. Resuming processing.")


# Collect all OFF file paths first to use with tqdm
off_files_to_process = []
for class_name in os.listdir(modelnet_path):
    class_path = os.path.join(modelnet_path, class_name)
    if os.path.isdir(class_path):
        if class_name not in class_to_idx:
            class_to_idx[class_name] = current_idx
            current_idx += 1
        label = class_to_idx[class_name]

        for split_type in ['train', 'test']:
            split_path = os.path.join(class_path, split_type)
            if os.path.isdir(split_path):
                for filename in os.listdir(split_path):
                    if filename.endswith('.off'):
                        off_files_to_process.append((os.path.join(split_path, filename), label))

# Process files with a progress bar
for off_filepath, label in tqdm(off_files_to_process, desc="Voxelizing Models"):
    voxel_grid = mesh_to_voxel_grid(off_filepath)
    if voxel_grid is not None:
        all_voxel_grids.append(voxel_grid)
        all_labels.append(label)

    # Save periodically to avoid data loss
    if len(all_voxel_grids) % 100 == 0 and len(all_voxel_grids) > 0:
        with open(processed_data_path, 'wb') as f:
            pickle.dump({
                'voxel_grids': all_voxel_grids,
                'labels': all_labels,
                'class_to_idx': class_to_idx,
                'current_idx': current_idx
            }, f)

# Final save after all processing is done
with open(processed_data_path, 'wb') as f:
    pickle.dump({
        'voxel_grids': all_voxel_grids,
        'labels': all_labels,
        'class_to_idx': class_to_idx,
        'current_idx': current_idx
    }, f)

print(f"Finished processing. Total voxel grids: {len(all_voxel_grids)}")
print(f"Total labels: {len(all_labels)}")
print(f"Detected classes: {class_to_idx}")

print(f"Loading processed data from {processed_data_path}...")
with open(processed_data_path, 'rb') as f:
    loaded_data = pickle.load(f)
    all_voxel_grids = loaded_data['voxel_grids']
    all_labels = loaded_data['labels']
    class_to_idx = loaded_data['class_to_idx']

print(f"Loaded {len(all_voxel_grids)} voxel grids and {len(all_labels)} labels.")

# Convert lists to NumPy arrays
all_voxel_grids = np.array(all_voxel_grids, dtype=np.float32)
all_labels = np.array(all_labels, dtype=np.int64)

# Define a custom PyTorch Dataset
class ModelNet10VoxelDataset(Dataset):
    def __init__(self, voxel_grids, labels):
        self.voxel_grids = torch.from_numpy(voxel_grids).unsqueeze(1) # Add channel dimension
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.voxel_grids[idx], self.labels[idx]

# Create the dataset instance
full_dataset = ModelNet10VoxelDataset(all_voxel_grids, all_labels)

# Define split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Calculate sizes for each split
total_size = len(full_dataset)
train_size = int(train_ratio * total_size)
val_size = int(val_ratio * total_size)
test_size = total_size - train_size - val_size # Ensure all samples are covered

# Split the dataset
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
