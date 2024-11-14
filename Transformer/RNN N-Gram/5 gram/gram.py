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

import re
from collections import Counter
from sklearn.model_selection import train_test_split

with open('Auguste_Maquet.txt', 'r') as f:
    text = f.read().lower()

sentences = re.split(r'[.!?]', text)  # Split text by punctuation marking the end of a sentence
sentences = [re.sub(r'[^a-z\s]', '', sentence).split() for sentence in sentences if sentence.strip()]  # Clean and split each sentence into words

words = [word for sentence in sentences for word in sentence]
word_count = Counter(words)
vocab = {word: i+1 for i, (word, count) in enumerate(word_count.items()) if count >= 1}
vocab['<UNK>'] = len(vocab) + 1

print(f'Vocabulary size: {len(vocab)}')

def tokenize_word(word):
    return vocab.get(word, vocab['<UNK>'])

tokenized_sentences = [[tokenize_word(word) for word in sentence] for sentence in sentences]

def generate_ngrams(tokenized_sentences, n=5):
    ngrams = []
    for sentence in tokenized_sentences:
        if len(sentence) > n:
            for i in range(len(sentence) - n):
                ngrams.append((sentence[i:i+n], sentence[i+n]))
    return ngrams

ngrams = generate_ngrams(tokenized_sentences)

train_data, test_data = train_test_split(ngrams, test_size=0.1, random_state=42)

print(f'Total n-grams: {len(ngrams)}')
print(f'Train data size: {len(train_data)}')
print(f'Test data size: {len(test_data)}')



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

vocab_size = len(vocab) + 1  # Including <UNK> token
embedding_dim = 100  # Word2Vec embeddings of 100 dimensions
model = NeuralLanguageModel(vocab_size, embedding_dim).to('mps')

print(model)

w2v_model = Word2Vec([words], vector_size=embedding_dim, min_count=1)
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in vocab.items():
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]
model.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))

# print(model.embedding.weight.data)


class NGramDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx][0]), torch.tensor(self.data[idx][1])


# print(train_data[:10])

train_dataset = NGramDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def calculate_perplexity(loss):
    return torch.exp(loss)

epochs = 25
for epoch in range(epochs):
    total_loss = 0
    model.train()
    for inputs, target in train_loader:
        inputs, target = inputs.to('mps'), target.to('mps')
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    perplexity = calculate_perplexity(torch.tensor(avg_loss))
    print(f'Epoch: {epoch+1}, Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}')

    model_path = "5gram_language_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


    model_path = "5gram_language_model.pth"
    model.load_state_dict(torch.load(model_path))

    model.eval()

    test_dataset = NGramDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    total_test_loss = 0

    with torch.no_grad():  # No need to compute gradients during testing
        for inputs, target in test_loader:
            inputs, target = inputs.to('mps'), target.to('mps')
            outputs = model(inputs)
            loss = criterion(outputs, target)
            total_test_loss += loss.item()

    avg_test_loss = total_test_loss / len(test_loader)

    test_perplexity = calculate_perplexity(torch.tensor(avg_test_loss))
    print(f'Test Loss: {avg_test_loss:.4f}, Test Perplexity: {test_perplexity:.4f}')


model_path = "5gram_language_model.pth"
model.load_state_dict(torch.load(model_path))
model.eval()
test_dataset = NGramDataset(test_data)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
total_test_loss = 0
def calculate_sentence_perplexity(outputs, target):
    log_probs = F.log_softmax(outputs, dim=-1)
    target_log_probs = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
    perplexity = torch.exp(-torch.mean(target_log_probs))
    return perplexity.item()
with open('test_perplexity_scores.txt', 'w') as f:
    total_test_loss = 0
    total_perplexity = 0
    sentence_count = 0
    with torch.no_grad():  
        for inputs, target in test_loader:
            inputs, target = inputs.to('mps'), target.to('mps')
            outputs = model(inputs)
            loss = criterion(outputs, target)
            total_test_loss += loss.item()
            for i, input_sentence in enumerate(inputs):
                sentence_tokens = [list(vocab.keys())[list(vocab.values()).index(token.item())] for token in input_sentence]
                sentence_str = ' '.join(sentence_tokens)
                sentence_perplexity = calculate_sentence_perplexity(outputs[i].unsqueeze(0), target[i].unsqueeze(0))
                total_perplexity += sentence_perplexity
                sentence_count += 1
                f.write(f'{sentence_str}\t{sentence_perplexity:.4f}\n')
    avg_test_loss = total_test_loss / len(test_loader)
    avg_perplexity = total_perplexity / sentence_count
    f.write(f'Average Perplexity\t{avg_perplexity:.4f}\n')
    print(f'Test Loss: {avg_test_loss:.4f}, Test Perplexity: {avg_perplexity:.4f}')