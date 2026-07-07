import sentencepiece as spm
import torch
import torch.nn as nn
import torch.optim as optim
import math
import time
from torch.utils.data import  DataLoader
from dataset import CoNaLaDataset
from layers import Encoder,Decoder
from train import train
from evaluate import evaluate
from inference import translate_sentence
from utils import epoch_time
from model import Seq2Seq

# ---  Initial Setup and SentencePiece Training ---

# Ensure reproducibility
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create a dummy text file for demonstration
dummy_nl_data = [
    "How to sum all elements in a list?",
    "Write a Python function to reverse a string.",
    "Calculate the factorial of a number iteratively.",
    "Find the maximum value in an array.",
    "Implement a binary search algorithm."
]
dummy_code_data = [
    "def sum_list(lst): return sum(lst)",
    "def reverse_string(s): return s[::-1]",
    "def factorial(n): res = 1; for i in range(1, n+1): res *= i; return res",
    "max_val = max(arr)",
    "def binary_search(arr, target): low = 0; high = len(arr) - 1; while low <= high: mid = (low + high) // 2; if arr[mid] == target: return mid; elif arr[mid] < target: low = mid + 1; else: high = mid - 1; return -1"
]

corpus_data = dummy_nl_data + dummy_code_data

corpus_file = "conala_corpus.txt"
with open(corpus_file, "w", encoding="utf-8") as f:
    for line in corpus_data:
        f.write(line + "\n")

print(f"Dummy corpus saved to {corpus_file}")

# Train a SentencePiece model
spm.SentencePieceTrainer.train(
    f'--input={corpus_file} --model_prefix=conala_spm --vocab_size=200 --character_coverage=1.0 --model_type=bpe'
)


# Load the trained SentencePiece model
sp = spm.SentencePieceProcessor()
sp.load("conala_spm.model")

# ---  Special Tokens and Data Preparation Helper Functions ---

UNK_IDX = sp.unk_id()
SOS_IDX = sp.bos_id()
EOS_IDX = sp.eos_id()

PAD_IDX = sp.pad_id()
if PAD_IDX == -1:
    print("SentencePiece model does not have a dedicated <pad> token. Using UNK_IDX for padding.")
    PAD_IDX = UNK_IDX

print(f"UNK_IDX: {UNK_IDX}")
print(f"SOS_IDX: {SOS_IDX}")
print(f"EOS_IDX: {EOS_IDX}")
print(f"PAD_IDX: {PAD_IDX}")

def tokenize_and_to_ids(text, sp_model, add_sos=False, add_eos=False):
    ids = sp_model.encode_as_ids(text)
    if add_sos:
        ids = [SOS_IDX] + ids
    if add_eos:
        ids = ids + [EOS_IDX]

tokenized_nl_ids = [tokenize_and_to_ids(nl, sp) for nl in dummy_nl_data]
tokenized_code_ids = [tokenize_and_to_ids(code, sp, add_sos=True, add_eos=True) for code in dummy_code_data]

conala_dataset = CoNaLaDataset(tokenized_nl_ids, tokenized_code_ids, PAD_IDX)

batch_size = 2
conala_dataloader = DataLoader(conala_dataset, batch_size=batch_size, shuffle=False)

print(f"\nCreated CoNaLaDataset with {len(conala_dataset)} samples.")
print(f"Max NL sequence length: {conala_dataset.max_nl_len}")
print(f"Max Code sequence length: {conala_dataset.max_code_len}")
print(f"DataLoader created with batch size: {batch_size}")

# --- Model Instantiation ---

INPUT_DIM = len(sp)
OUTPUT_DIM = len(sp)
EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
DROPOUT = 0.5

enc = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
dec = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)

model = Seq2Seq(enc, dec, device).to(device)

print(f"The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")

# ---  Loss Function and Optimizer ---

TRG_PAD_IDX = PAD_IDX
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)
optimizer = optim.Adam(model.parameters())


# --- 10. Training Loop ---

N_EPOCHS = 10
CLIP = 1.0

for epoch in range(N_EPOCHS):
    start_time = time.time()

    train_loss = train(model, conala_dataloader, optimizer, criterion, CLIP, device)
    # For this dummy data, we'll evaluate on the same data for demonstration
    valid_loss = evaluate(model, conala_dataloader, criterion, device)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):.3f}')

print("\nTraining complete.")

# ---  Inference Examples ---

print("\n--- Inference Examples ---")

# Use the max_code_len from the dataset for max_len during inference
inference_max_len = conala_dataset.max_code_len + 5 # Add a small buffer

sample_nl_queries = [
    "How to sum all elements in a list?",
    "reverse a string in python",
    "calculate factorial",
    "find maximum in array",
    "implement binary search"
]

for query in sample_nl_queries:
    predicted_code = translate_sentence(query, model, sp, device, inference_max_len)
    print(f"NL Query: '{query}'")
    print(f"Predicted Code: '{predicted_code}'")
    print("--------------------------------------------------")
