import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import re
from collections import Counter
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
import numpy as np
from torch.nn.utils.rnn import pad_sequence

with open('Auguste_Maquet.txt', 'r') as f:
    text = f.read().lower()

sentences = re.split(r'[.!?]', text)  
sentences = [re.sub(r'[^a-z\s]', '', sentence).split() for sentence in sentences if sentence.strip()]  

words = [word for sentence in sentences for word in sentence]
word_count = Counter(words)
vocab = {word: i+1 for i, (word, count) in enumerate(word_count.items()) if count >= 1}
vocab['<UNK>'] = len(vocab) + 1

def tokenize_word(word):
    return vocab.get(word, vocab['<UNK>'])

tokenized_sentences = [[tokenize_word(word) for word in sentence] for sentence in sentences]

tokenized_sentences = [sentence for sentence in tokenized_sentences if len(sentence) <= 20]

train_data, test_data = train_test_split(tokenized_sentences, test_size=0.1, random_state=42)

class SentenceDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sentence = torch.tensor(self.data[idx][:-1], dtype=torch.long)  
        target = torch.tensor(self.data[idx][1:], dtype=torch.long) 
        return sentence, target

def collate_fn(batch):
    sentences, targets = zip(*batch)
    sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
    return sentences_padded, targets_padded

class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1):
        super(LSTMLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)
        output = self.fc(lstm_out)
        return output, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, hidden_dim).to('mps'),
                torch.zeros(1, batch_size, hidden_dim).to('mps'))

vocab_size = len(vocab) + 1
embedding_dim = 100
hidden_dim = 300
model = LSTMLanguageModel(vocab_size, embedding_dim, hidden_dim).to('mps')

w2v_model = Word2Vec([words], vector_size=embedding_dim, min_count=1)
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in vocab.items():
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]
model.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))

train_dataset = SentenceDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding index during loss calculation
optimizer = optim.Adam(model.parameters(), lr=0.001)

def calculate_perplexity(loss):
    return torch.exp(loss)

epochs = 25
for epoch in range(epochs):
    total_loss = 0
    model.train()
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to('mps'), targets.to('mps')
        batch_size = inputs.size(0)
        
        hidden = model.init_hidden(batch_size)
        optimizer.zero_grad()
        
        outputs, hidden = model(inputs, hidden)
        outputs = outputs.view(-1, vocab_size)
        targets = targets.view(-1)
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    perplexity = calculate_perplexity(torch.tensor(avg_loss))
    print(f'Epoch: {epoch+1}, Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}')

    model_path = "lstm_language_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")



    model_path = "lstm_language_model.pth"
    model.load_state_dict(torch.load(model_path))

    model.eval()

    test_dataset = SentenceDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

    total_loss = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to('mps'), targets.to('mps')
            batch_size = inputs.size(0)
        
            hidden = model.init_hidden(batch_size)
        
            outputs, hidden = model(inputs, hidden)
            outputs = outputs.view(-1, vocab_size)
            targets = targets.view(-1)
        
            loss = criterion(outputs, targets)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)

    test_perplexity = calculate_perplexity(torch.tensor(avg_loss))
    print(f'Test Loss: {avg_loss:.4f}, Test Perplexity: {test_perplexity:.4f}')


model_path = "lstm_language_model.pth"
model.load_state_dict(torch.load(model_path))
model.eval()
test_dataset = SentenceDataset(test_data)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
sentence_perplexities = []

output_file = "perplexity_scores.txt"
with open(output_file, 'w') as f_out:
    total_loss = 0
    total_sentences = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to('mps'), targets.to('mps')
            batch_size = inputs.size(0)
            hidden = model.init_hidden(batch_size)
            outputs, hidden = model(inputs, hidden)
            outputs = outputs.view(-1, vocab_size)
            targets = targets.view(-1)
            loss = criterion(outputs, targets)
            perplexity = calculate_perplexity(loss).item()
            for i, input_sentence in enumerate(inputs):
                sentence = ' '.join([list(vocab.keys())[list(vocab.values()).index(token.item())] if token.item() in vocab.values() else '<UNK>' for token in input_sentence if token.item() != 0])  # Remove padding tokens
                f_out.write(f"{sentence}\t{perplexity:.4f}\n")
                total_sentences += 1
            total_loss += loss.item()
    avg_loss = total_loss / len(test_loader)
    avg_perplexity = calculate_perplexity(torch.tensor(avg_loss)).item()
    f_out.write(f"Average Perplexity Score: {avg_perplexity:.4f}\n")
