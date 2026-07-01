import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import torch
from dataset import BirdclefDataset
from model import VGGish
from train import train
from torch.utils.data import DataLoader
import torch.nn as nn
import os

api=KaggleApi()
api.authenticate()

# Configuration parameters for VGGish preprocessing
SAMPLE_RATE = 16000
N_FFT = 400  # Corresponds to 25ms window size at 16kHz
WIN_LENGTH = 400
HOP_LENGTH = 160 # Corresponds to 10ms hop size at 16kHz
N_MELS = 64

path = api.dataset_download_files("vbs2004/birdclef-audio-dataset")

print("Path to dataset files:",path)

# Construct the full path to the preprocessed file
preprocessed_file_path = os.path.join(path, 'birdclef_preprocessed.pt')

print(f"Attempting to load: {preprocessed_file_path}")

try:
    # Load the preprocessed data
    preprocessed_data = torch.load(preprocessed_file_path)

    print(f"Successfully loaded data.")
    print(f"Type of loaded data: {type(preprocessed_data)}")

    # If it's a tensor, print its shape
    if isinstance(preprocessed_data, torch.Tensor):
        print(f"Shape of preprocessed data tensor: {preprocessed_data.shape}")
    # If it's a dictionary or list, print keys/first few elements
    elif isinstance(preprocessed_data, dict):
        print(f"Keys in loaded dictionary: {preprocessed_data.keys()}")
    elif isinstance(preprocessed_data, list):
        print(f"Number of items in loaded list: {len(preprocessed_data)}")
        if len(preprocessed_data) > 0 and isinstance(preprocessed_data[0], torch.Tensor):
            print(f"Shape of first item (if tensor): {preprocessed_data[0].shape}")

except Exception as e:
    print(f"Error loading preprocessed data: {e}")

if isinstance(preprocessed_data, dict):
    signals = preprocessed_data['signals']
    labels = preprocessed_data['labels']

    print(f"\nShape of 'signals' tensor: {signals.shape}")
    print(f"Type of 'signals' tensor: {signals.dtype}")
    print(f"\nShape of 'labels' tensor: {labels.shape}")
    print(f"Type of 'labels' tensor: {labels.dtype}")

    # Check if 'signals' matches expected VGGish input shape (batch, 1, 96, 64)
    # The first dimension would be the batch size (number of samples)
    # The remaining (1, 96, 64) is the expected input for a single VGGish patch
    expected_vggish_input_dims = (1, 96, 64)
    if signals.ndim == 4 and signals.shape[1:] == expected_vggish_input_dims:
        print("\n'signals' tensor shape is compatible with VGGish input (batch, 1, 96, 64).")
    else:
        print(f"\nWarning: 'signals' tensor shape {signals.shape} does not directly match expected VGGish input shape (batch, 1, 96, 64). Further reshaping or processing might be needed.")

    # Display first few labels for context
    print(f"\nFirst 10 labels: {labels[:10].tolist()}")
else:
    print("Error: Loaded data is not a dictionary. Cannot extract 'signals' and 'labels'.")

birdclef_dataset = BirdclefDataset(signals, labels)


# 2. PyTorch DataLoader
batch_size = 32 # You can adjust this
dataloader = DataLoader(birdclef_dataset, batch_size=batch_size, shuffle=True)

print(f"Number of samples in dataset: {len(birdclef_dataset)}")
print(f"Number of batches per epoch: {len(dataloader)}")

# 3. Classification Head
class ClassificationHead(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

# Get the number of unique classes from the labels
num_classes = len(torch.unique(labels))
embedding_dim = 128 # VGGish outputs 128-dim embeddings

# Instantiate the VGGish model and Classification Head
vggish_model = VGGish() # Assuming VGGish class is defined and available
classification_head = ClassificationHead(embedding_dim, num_classes)

# Combine VGGish with the classification head
class FullModel(nn.Module):
    def __init__(self, vggish_base, classification_head):
        super(FullModel, self).__init__()
        self.vggish_base = vggish_base
        self.classification_head = classification_head

    def forward(self, x):
        embeddings = self.vggish_base(x)
        logits = self.classification_head(embeddings)
        return logits

model = FullModel(vggish_model, classification_head)

# 4. Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5 # You can adjust this

train(num_epochs,model,dataloader,optimizer,criterion,device)


print("\n--- Model Inference and Visualization ---")

# Put the model in evaluation mode
model.eval()

total_correct = 0
total_processed = 0

all_labels = []
all_predicted = []

print("\nPerforming inference on a larger sample for visualization...")

with torch.no_grad(): # Disable gradient calculations for inference
    for i, (inputs, labels) in enumerate(dataloader):
        if i >= 50: # Process 50 batches for a more substantial confusion matrix
            break

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        total_processed += labels.size(0)
        total_correct += (predicted == labels).sum().item()

        all_labels.extend(labels.cpu().numpy())
        all_predicted.extend(predicted.cpu().numpy())

accuracy = (total_correct / total_processed) * 100 if total_processed > 0 else 0
print(f"\nAccuracy on {total_processed} samples: {accuracy:.2f}%")

# 1. Confusion Matrix
print("\nGenerating Confusion Matrix...")
conf_matrix = confusion_matrix(all_labels, all_predicted)

# Due to potentially many classes, showing all might be too dense.
# We can visualize a subset or a normalized version.
# For now, let's visualize the raw confusion matrix.

plt.figure(figsize=(20, 18)) # Adjust size for better visibility, especially with many classes
sns.heatmap(conf_matrix, annot=False, cmap='Blues', fmt='d') # annot=False for many classes
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Confusion Matrix (First {total_processed} samples)')
plt.show()

# Optional: Further visualization ideas
# 2. t-SNE/UMAP of embeddings (requires collecting embeddings during inference)
# You would need to modify the FullModel to output embeddings before the classification head
# and collect them during inference to then visualize.