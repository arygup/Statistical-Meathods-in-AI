import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import Counter
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset

# Tokenization functions for English and French
def tokenize_english(sentence):
    return [word.lower() for word in word_tokenize(sentence, language='english')]

def tokenize_french(sentence):
    return [word.lower() for word in word_tokenize(sentence, language='french')]

# Function to read data from files and tokenize
def read_and_tokenize(file_path, tokenizer):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [tokenizer(line.strip()) for line in file.readlines()]

# Function to build vocabulary from tokenized sentences
def build_vocabulary(tokenized_sentences, min_frequency=1):
    words = [word for sentence in tokenized_sentences for word in sentence]
    word_count = Counter(words)
    vocab = {word: idx + 4 for idx, (word, count) in enumerate(word_count.items()) if count >= min_frequency}

    # Add special tokens to vocabulary
    vocab.update({
        '<unk>': 0,
        '<pad>': 1,
        '<sos>': 2,
        '<eos>': 3
    })
    return vocab

# Function to convert tokens to indices
def convert_tokens_to_indices(sentences, vocab):
    return [
        [vocab['<sos>']] + [vocab.get(token, vocab['<unk>']) for token in sentence] + [vocab['<eos>']]
        for sentence in sentences
    ]

# Function to pad sentences to a maximum length
def pad_sentences_to_length(token_indices, pad_idx, max_length):
    return [seq + [pad_idx] * (max_length - len(seq)) for seq in token_indices]

# Custom dataset class for translation
class TranslationDataset(Dataset):
    def __init__(self, source_data, target_data):
        self.source_data = source_data
        self.target_data = target_data

    def __len__(self):
        return len(self.source_data)

    def __getitem__(self, idx):
        return torch.tensor(self.source_data[idx]), torch.tensor(self.target_data[idx])

# Feed Forward Neural Network used in Transformer
class FeedForwardNN(nn.Module):
    def __init__(self, context_dim: int, compression_factor: float):
        super(FeedForwardNN, self).__init__()
        hidden_dim = int(context_dim * compression_factor)
        self.fc1 = nn.Linear(context_dim, hidden_dim)  # Compress down
        self.fc2 = nn.Linear(hidden_dim, context_dim)  # Expand back up
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.gelu(self.fc1(x))
        x = self.fc2(x)
        return x

# Multi-Head Attention Mechanism
class MultiHeadAttention(nn.Module):
    def __init__(self, context_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = context_dim // num_heads

        assert context_dim % num_heads == 0, "context_dim must be divisible by num_heads"

        self.W_q = nn.Linear(context_dim, context_dim, bias=False)
        self.W_k = nn.Linear(context_dim, context_dim, bias=False)
        self.W_v = nn.Linear(context_dim, context_dim, bias=False)
        self.W_o = nn.Linear(context_dim, context_dim)

    def split_head(self, x):
        batch_size, seq_len, dim = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_head(self.W_q(Q))
        K = self.split_head(self.W_k(K))
        V = self.split_head(self.W_v(V))

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn = torch.matmul(attn_weights, V)

        attn = attn.transpose(1, 2).contiguous().view(Q.size(0), -1, self.context_dim)
        output = self.W_o(attn)
        return output
