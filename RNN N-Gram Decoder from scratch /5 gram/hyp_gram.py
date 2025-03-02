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
import matplotlib.pyplot as plt
import pandas as pd

# Load and clean the dataset
with open('Auguste_Maquet copy.txt', 'r') as f:
    text = f.read().lower()

# Clean the text - removing special characters
text = re.sub(r'[^a-z\s]', '', text)
words = text.split()

# Create a vocabulary
word_count = Counter(words)
vocab = {word: i+1 for i, (word, count) in enumerate(word_count.items()) if count >= 1}
vocab['<UNK>'] = len(vocab) + 1

def tokenize_word(word):
    return vocab.get(word, vocab['<UNK>'])

tokenized_text = [tokenize_word(word) for word in words]

# Generate 5-grams for training
def generate_ngrams(tokenized_text, n=5):
    ngrams = []
    for i in range(len(tokenized_text) - n):
        ngrams.append((tokenized_text[i:i+n], tokenized_text[i+n]))
    return ngrams

ngrams = generate_ngrams(tokenized_text)
train_data, test_data = train_test_split(ngrams, test_size=0.1, random_state=42)

# Dataset and DataLoader classes
class NGramDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx][0]), torch.tensor(self.data[idx][1])

train_dataset = NGramDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = NGramDataset(test_data)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the Neural Language Model
class NeuralLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim1, hidden_dim2, dropout_rate):
        super(NeuralLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 5, hidden_dim1)  # 5-gram concatenation
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_dim2, vocab_size)
    
    def forward(self, x):
        embedded = self.embedding(x).view(x.size(0), -1)  # Flatten the 5-grams
        hidden1 = F.relu(self.fc1(embedded))
        hidden1 = self.dropout1(hidden1)
        hidden2 = F.relu(self.fc2(hidden1))
        hidden2 = self.dropout2(hidden2)
        output = self.fc3(hidden2)
        return output

# Word2Vec initialization
w2v_model = Word2Vec([words], vector_size=100, min_count=1)
vocab_size = len(vocab) + 1  # Including <UNK> token
embedding_dim = 100

def initialize_embeddings(model, vocab, w2v_model):
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in vocab.items():
        if word in w2v_model.wv:
            embedding_matrix[i] = w2v_model.wv[word]
    model.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))

# Train and evaluate model
def train_and_evaluate_model(train_loader, test_loader, model, criterion, optimizer, epochs=3, device='mps'):
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for inputs, target in train_loader:
            inputs, target = inputs.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Evaluate on test data
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for inputs, target in test_loader:
                inputs, target = inputs.to(device), target.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, target)
                total_test_loss += loss.item()
        
        avg_test_loss = total_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')
        
    train_perplexity = [torch.exp(torch.tensor(loss)) for loss in train_losses]
    test_perplexity = [torch.exp(torch.tensor(loss)) for loss in test_losses]
    
    return train_perplexity, test_perplexity

# Hyperparameter sweeps
dropout_rates = [0.0, 0.3, 0.5]
hidden_dims = [(300, 300), (200, 200), (400, 400)]
optimizers = ['Adam', 'SGD', 'RMSprop']

results = []

for dropout in dropout_rates:
    for hidden_dim1, hidden_dim2 in hidden_dims:
        for opt_name in optimizers:
            # Initialize model
            model = NeuralLanguageModel(vocab_size, embedding_dim, hidden_dim1, hidden_dim2, dropout).to('mps')
            initialize_embeddings(model, vocab, w2v_model)
            
            # Initialize optimizer
            if opt_name == 'Adam':
                optimizer = optim.Adam(model.parameters(), lr=0.001)
            elif opt_name == 'SGD':
                optimizer = optim.SGD(model.parameters(), lr=0.01)
            elif opt_name == 'RMSprop':
                optimizer = optim.RMSprop(model.parameters(), lr=0.001)
            
            # Initialize loss function
            criterion = nn.CrossEntropyLoss()

            # Train and evaluate
            train_perplexity, test_perplexity = train_and_evaluate_model(train_loader, test_loader, model, criterion, optimizer)

            # Record results
            results.append({
                'dropout': dropout,
                'hidden_dims': (hidden_dim1, hidden_dim2),
                'optimizer': opt_name,
                'train_perplexity': train_perplexity,
                'test_perplexity': test_perplexity
            })
            
            # Save the model for this configuration
            # model_path = f"model_{hidden_dim1}_{hidden_dim2}_dropout{dropout}_{opt_name}.pth"
            # torch.save(model.state_dict(), model_path)
            # print(f"Model saved to {model_path}")

# Save results to CSV
csv_data = []
for result in results:
    for epoch, (train_ppl, test_ppl) in enumerate(zip(result['train_perplexity'], result['test_perplexity'])):
        csv_data.append({
            'Dropout': result['dropout'],
            'Hidden_Dims': result['hidden_dims'],
            'Optimizer': result['optimizer'],
            'Epoch': epoch + 1,
            'Train_Perplexity': train_ppl.item(),
            'Test_Perplexity': test_ppl.item()
        })
df = pd.DataFrame(csv_data)
df.to_csv('perplexity_results.csv', index=False)
print("Results saved to perplexity_results.csv")

# Plot and save the selected results (subset for clarity)
for result in results:
    if result['dropout'] == 0.5 and result['optimizer'] == 'Adam':  # Plot only selected configurations
        plt.figure(figsize=(8, 6))
        plt.plot(result['train_perplexity'], label='Train Perplexity')
        plt.plot(result['test_perplexity'], label='Test Perplexity')
        plt.title(f"Dropout: {result['dropout']}, Hidden: {result['hidden_dims']}, Optimizer: {result['optimizer']}")
        plt.xlabel("Epochs")
        plt.ylabel("Perplexity")
        plt.legend()

        # Save the plot
        plot_path = f"perplexity_plot_{result['hidden_dims']}_dropout{result['dropout']}_{result['optimizer']}.png"
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
        plt.close()
