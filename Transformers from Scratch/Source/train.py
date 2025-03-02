import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import math
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from collections import Counter
from encoder import Encoder
from decoder import Decoder
from utils import (
    tokenize_english, tokenize_french, read_and_tokenize, build_vocabulary,
    convert_tokens_to_indices, pad_sentences_to_length, TranslationDataset
)

import pickle



nltk.download('punkt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# device = torch.device('mps')
# print(f"Using device: {device}")

# max_allowed_length = 3

max_allowed_length = 100


file_paths = {
    'train_en': 'train.en',
    'train_fr': 'train.fr',
    'dev_en': 'dev.en',
    'dev_fr': 'dev.fr'
}

# Preprocessing the data for training
train_en_sentences = read_and_tokenize(file_paths['train_en'], tokenize_english)
train_fr_sentences = read_and_tokenize(file_paths['train_fr'], tokenize_french)

# Filter sentences based on length
filtered_en_sentences = [s for s in train_en_sentences if len(s) <= max_allowed_length]
filtered_fr_sentences = [
    train_fr_sentences[i] for i in range(len(train_en_sentences))
    if len(train_en_sentences[i]) <= max_allowed_length
]

assert len(filtered_en_sentences) == len(filtered_fr_sentences), "Mismatch in sentence counts."

# Build vocabularies
en_vocab = build_vocabulary(filtered_en_sentences)
fr_vocab = build_vocabulary(filtered_fr_sentences)

with open('en_vocab.pkl', 'wb') as f:
    pickle.dump(en_vocab, f)
with open('fr_vocab.pkl', 'wb') as f:
    pickle.dump(fr_vocab, f)

# Convert tokens to indices
train_en_indices = convert_tokens_to_indices(filtered_en_sentences, en_vocab)
train_fr_indices = convert_tokens_to_indices(filtered_fr_sentences, fr_vocab)

# Find maximum lengths
max_len_en = max(len(seq) for seq in train_en_indices)
max_len_fr = max(len(seq) for seq in train_fr_indices)

# Pad sequences
train_en_padded = pad_sentences_to_length(train_en_indices, en_vocab['<pad>'], max_len_en)
train_fr_padded = pad_sentences_to_length(train_fr_indices, fr_vocab['<pad>'], max_len_fr)

# Create DataLoader
train_dataset = TranslationDataset(train_en_padded, train_fr_padded)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

print(f"Training data prepared. English sentences: {len(train_en_padded)}, French sentences: {len(train_fr_padded)}")

# Define the Transformer model
class Transformer(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size, context_dim, num_heads, num_layers, compression_factor, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.enc_emb = nn.Embedding(source_vocab_size, context_dim)
        self.dec_emb = nn.Embedding(target_vocab_size, context_dim)

        self.pe = self.create_positional_encoding(max_seq_length, context_dim).to(device)
        self.dropout = nn.Dropout(dropout)

        self.encoders = nn.ModuleList([
            Encoder(context_dim, num_heads, compression_factor, dropout) for _ in range(num_layers)
        ])
        self.decoders = nn.ModuleList([
            Decoder(context_dim, num_heads, compression_factor, dropout) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(context_dim, target_vocab_size)

    def create_positional_encoding(self, max_seq_length, context_dim):
        pe = torch.zeros(max_seq_length, context_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, context_dim, 2) * (-math.log(10000.0) / context_dim))

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        return pe

    def add_positional_encoding(self, x):
        seq_len = x.size(1)
        pe = self.pe[:seq_len, :].unsqueeze(0).to(x.device)
        return x + pe

    def forward(self, source, target):
        source_mask = (source != en_vocab['<pad>']).unsqueeze(1).unsqueeze(2)
        target_mask = (target != fr_vocab['<pad>']).unsqueeze(1).unsqueeze(2)
        seq_len = target.size(1)
        nopeak_mask = torch.tril(torch.ones((1, 1, seq_len, seq_len), device=target.device)).bool()
        target_mask = target_mask & nopeak_mask

        # Embeddings
        source_embedded = self.dropout(self.add_positional_encoding(self.enc_emb(source)))
        target_embedded = self.dropout(self.add_positional_encoding(self.dec_emb(target)))

        # Encoder
        enc_output = source_embedded
        for encoder in self.encoders:
            enc_output = encoder(enc_output, source_mask)

        # Decoder
        dec_output = target_embedded
        for decoder in self.decoders:
            dec_output = decoder(dec_output, enc_output, source_mask, target_mask)

        output = self.fc_out(dec_output)
        return output

# Training functions
def train_one_epoch(model, loader, criterion, optimizer, scheduler, epoch):
    model.train()
    total_loss = 0
    for src, tgt in tqdm(loader, desc=f"Epoch {epoch+1}"):
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output = model(src, tgt[:, :-1])
        loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
    return avg_loss

# Instantiate the model
num_epochs = 1
model = Transformer(
    source_vocab_size=len(en_vocab),
    target_vocab_size=len(fr_vocab),
    context_dim=512,
    num_heads=4,
    num_layers=4,
    compression_factor=4,
    max_seq_length=max(max_len_en, max_len_fr),
    dropout=0.3
).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=fr_vocab['<pad>'])
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=5e-4, steps_per_epoch=len(train_loader), epochs=num_epochs
)

# Training loop
for epoch in range(num_epochs):
    train_one_epoch(model, train_loader, criterion, optimizer, scheduler, epoch)
    # Save the model checkpoint

torch.save(model.state_dict(), f"transformer.pt")
