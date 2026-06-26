import torch
from torch.utils.data import DataLoader
from dataset import train_dataset,val_dataset,vocab_size,encode,decode
from model import RetNet
import torch.nn as nn
from train import train
from inference import generate_text
batch_size = 128 # A common batch size
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model hyperparameters
hidden_dim = 512
num_layers = 6
num_heads = 8
head_dim = hidden_dim // num_heads # Ensure hidden_dim is divisible by num_heads
intermediate_dim = hidden_dim * 4 # Typically 4x hidden_dim for FFN
dropout = 0.1

model = RetNet(
    vocab_size=vocab_size,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    num_heads=num_heads,
    head_dim=head_dim,
    intermediate_dim=intermediate_dim,
    dropout=dropout
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4) # Learning rate can be tuned
loss_fn = nn.CrossEntropyLoss(ignore_index=model.pad_token_id) # Ignore padding token if used

epochs = 1# Number of training epochs, can be tuned

train(epochs,model,train_dataloader,val_dataloader,loss_fn,optimizer,device)

# Define an example prompt
prompt = "ROMEO:"

# Set the maximum number of new tokens to generate
max_new_tokens = 200

# Ensure the model is on CPU for generation as specified
model.to('cpu')

print(f"Generating text with prompt: '{prompt}' and max_new_tokens: {max_new_tokens}")

# Call the generate_text function
generated_text = generate_text(
    model=model,
    prompt=prompt,
    max_new_tokens=max_new_tokens,
    encode=encode,
    decode=decode,
    device='cpu' # Explicitly use CPU for generation
)

# Print the generated text
print("\n--- Generated Text ---")
print(generated_text)
print("----------------------")

# Move the model back to the original device if it was on GPU for further training/evaluation
model.to(device)
