import torch
import torch.nn as nn
import re
from collections import Counter
from gensim.models import Word2Vec
import numpy as np

class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim=300):
        super(LSTMLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, hidden, cell):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        lstm_out = lstm_out[:, -1, :]  # Get the output for the last time step
        output = self.fc(lstm_out)
        return output, hidden, cell

def load_vocab(file_path):
    with open(file_path, 'r') as f:
        text = f.read().lower()

    sentences = re.split(r'[.!?]', text)
    sentences = [re.sub(r'[^a-z\s]', '', sentence).split() for sentence in sentences if sentence.strip()]

    words = [word for sentence in sentences for word in sentence]
    word_count = Counter(words)
    vocab = {word: i+1 for i, (word, count) in enumerate(word_count.items()) if count >= 1}
    vocab['<UNK>'] = len(vocab) + 1
    return vocab

vocab = load_vocab('Auguste_Maquet.txt')
vocab_size = len(vocab) + 1

embedding_dim = 100
hidden_dim = 300
model = LSTMLanguageModel(vocab_size, embedding_dim, hidden_dim).to('mps')

model_path = "lstm_language_model.pth"
model.load_state_dict(torch.load(model_path))

def init_hidden(batch_size=1):
    return torch.zeros(1, batch_size, hidden_dim).to('mps'), torch.zeros(1, batch_size, hidden_dim).to('mps')

reverse_vocab = {idx: word for word, idx in vocab.items()}

def tokenize_word(word):
    return vocab.get(word, vocab['<UNK>'])

def predict_next_word(model, input_word, hidden, cell):
    token = torch.tensor([[tokenize_word(input_word)]]).to('mps')
    output, hidden, cell = model(token, hidden, cell)
    predicted_idx = torch.argmax(output, dim=-1).item()
    return reverse_vocab.get(predicted_idx, '<UNK>'), hidden, cell

def generate_sentence(start_word, max_len=20):
    model.eval()
    hidden, cell = init_hidden()
    current_word = start_word
    sentence = [current_word]
    
    for _ in range(max_len - 1):  # Limit sentence to 20 words
        next_word, hidden, cell = predict_next_word(model, current_word, hidden, cell)
        sentence.append(next_word)
        if next_word == '.':  # End the sentence when a period is predicted
            break
        current_word = next_word
    
    return ' '.join(sentence)

start_word = "to"
print(f"Starting word: {start_word}")
generated_sentence = generate_sentence(start_word)
print(f"Generated sentence: {generated_sentence}")
