import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import re
from collections import Counter
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
import numpy as np

class NeuralLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim1=300, hidden_dim2=300):
        super(NeuralLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 5, hidden_dim1)  # 5-gram concatenation
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        embedded = self.embedding(x).view(x.size(0), -1)  # Flatten the 5-grams
        hidden1 = F.relu(self.fc1(embedded))
        hidden2 = F.relu(self.fc2(hidden1))
        output = self.fc3(hidden2)
        return output

vocab_size = 25107  # Including <UNK> token
embedding_dim = 100
model_path = "5gram_language_model.pth"

model = NeuralLanguageModel(vocab_size, embedding_dim)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()  

with open('Auguste_Maquet.txt', 'r') as f:
    text = f.read().lower()

text = re.sub(r'[^a-z\s]', '', text)
words = text.split()

# print(words[:10])

word_count = Counter(words)
vocab = {word: i+1 for i, (word, count) in enumerate(word_count.items()) if count >= 1}
vocab['<UNK>'] = len(vocab) + 1


inv_vocab = {v: k for k, v in vocab.items()}

def tokenize_input(input_words):
    return [vocab.get(word, vocab['<UNK>']) for word in input_words]

def predict_next_word(input_words):
    model.eval()
    tokenized_input = torch.tensor([tokenize_input(input_words)]).long()
    with torch.no_grad():
        output = model(tokenized_input)
        predicted_word_idx = torch.argmax(output, dim=-1).item()
    return inv_vocab.get(predicted_word_idx, '<UNK>')

def predict_sentence(initial_words, num_predictions=10):
    sentence = initial_words
    for _ in range(num_predictions):
        next_word = predict_next_word(sentence[-5:])  
        sentence.append(next_word)
    return ' '.join(sentence)

initial_words = ['lord', 'says', 'that', 'you', 'must']  
print(f"Initial Words: {initial_words}")
predicted_sentence = predict_sentence(initial_words, num_predictions=10)
print(f"Predicted Sentence: {predicted_sentence}")
