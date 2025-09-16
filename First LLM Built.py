import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import json
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter

nltk.download('punkt')

# --- Model Definition ---
class FirstLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(lstm_out)
        return output

# --- Hyperparameters ---
vocab_size = 10000
embedding_dim = 256
hidden_dim = 512
num_layers = 2
seq_length = 20
batch_size = 32
num_epochs = 10
learning_rate = 0.001

# --- Data Loading and Preprocessing ---

# Download the NLTK tokenizer model if not already present
# Download the NLTK tokenizer model if not already present
print("Checking for NLTK's 'punkt' model...")
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK's 'punkt' model...")
    nltk.download('punkt', quiet=True)
    print("Download complete.")

# Load raw data from the .ndjson file
sequences_data = []
with open('C:\\Users\\doubl\\Downloads\\people_0.ndjson', 'r', encoding='utf-8') as f:
    for line in f:
        sequences_data.append(json.loads(line))

# 1. Extract text and tokenize
print("Tokenizing text from 'abstract' field...")
all_tokens = []
for item in sequences_data:
    abstract_text = item.get('abstract')
    if abstract_text and isinstance(abstract_text, str):
        all_tokens.extend(word_tokenize(abstract_text.lower()))

# 2. Build vocabulary
print("Building vocabulary...")
word_counts = Counter(all_tokens)
# vocab_size-2 to leave space for <pad> and <unk> tokens
most_common_words = [word for word, count in word_counts.most_common(vocab_size - 2)]
word_to_idx = {word: i+2 for i, word in enumerate(most_common_words)}
word_to_idx['<pad>'] = 0 # Padding token
word_to_idx['<unk>'] = 1 # Unknown word token

# 3. Convert all abstracts to token IDs
print("Converting text to token IDs...")
tokenized_sequences = []
for item in sequences_data:
    abstract_text = item.get('abstract')
    if abstract_text and isinstance(abstract_text, str):
        tokens = word_tokenize(abstract_text.lower())
        token_ids = [word_to_idx.get(word, 1) for word in tokens] # Use 1 (<unk>) for unknown words
        tokenized_sequences.append(token_ids)

# 4. Create input and target sequences
print("Creating input/target sequences...")
inputs = []
targets = []
for token_ids in tokenized_sequences:
    if len(token_ids) >= seq_length + 1:
        for i in range(len(token_ids) - seq_length):
            inputs.append(torch.tensor(token_ids[i:i+seq_length], dtype=torch.long))
            targets.append(torch.tensor(token_ids[i+1:i+seq_length+1], dtype=torch.long))

# Check if any valid sequences were created
if not inputs:
    raise ValueError("No sequences of sufficient length were created. Check your data or seq_length.")

inputs = torch.stack(inputs)
targets = torch.stack(targets)

print(f"Created {len(inputs)} sequences.")

# --- DataLoader ---
dataset = TensorDataset(inputs, targets)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- Model, Loss, and Optimizer ---

# Add this section to detect and select the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

model = FirstLanguageModel(vocab_size, embedding_dim, hidden_dim, num_layers)
model.to(device) # Move the model's parameters to the GPU

print("\nModel Architecture:")
print(model)

criterion = nn.CrossEntropyLoss(ignore_index=word_to_idx['<pad>']) # Ignore padding in loss calculation
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# --- Training Loop ---
print("\nStarting training...")
for epoch in range(num_epochs):
    # Add a tracker for total loss in the epoch
    epoch_loss = 0
    num_batches = 0
    for batch_inputs, batch_targets in dataloader:
        # Move the data for the current batch to the GPU
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)

        optimizer.zero_grad()
        outputs = model(batch_inputs)
        
        # Reshape for CrossEntropyLoss
        outputs_flat = outputs.view(-1, vocab_size)
        targets_flat = batch_targets.view(-1)
        
        loss = criterion(outputs_flat, targets_flat)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1
        
    # Print average loss for the epoch
    avg_epoch_loss = epoch_loss / num_batches
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")

print("\nTraining complete.")