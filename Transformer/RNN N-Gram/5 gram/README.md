
# Observations
![image](output.png)
### 1. Dropout
- **Observation:** The violin plots for dropout rates show a general trend where lower dropout rates (e.g., 0.0 and 0.1) tend to have lower central values of perplexity. 
- **Interpretation:** Lower dropout rates may be better for this specific model and dataset because they allow more features from the neurons to be retained during training, which can lead to better learning with less information loss. However, it is important to ensure that this does not lead to overfitting, especially in scenarios where the model is complex or the dataset is small.

### 2. Hidden Dimensions
- **Observation:** The model configuration with hidden dimensions of `(200, 200)` tends to show lower perplexity compared to larger dimensions such as `(300, 300)` and `(400, 400)`.
- **Interpretation:** Smaller hidden dimensions might be providing a sufficient capacity to model the necessary relationships in the data without overly complexifying the model, which can lead to overfitting and higher perplexity in testing. It suggests a balanced approach to model complexity might be more effective.

### 3. Optimizer
- **Observation:** Adam optimizer shows a generally lower spread and lower central value of perplexity compared to SGD and RMSprop.
- **Interpretation:** Adam is known for combining the benefits of two other extensions of stochastic gradient descent, specifically Adaptive Gradient Algorithm (AdaGrad) and Root Mean Square Propagation (RMSProp), which help in handling sparse gradients on noisy problems. It is likely providing a more efficient and stable convergence in this case, leading to lower perplexity.

### Conclusion
The best model configuration based on the provided data and perplexity metrics appears to be:
- **Dropout Rate:** Low (around 0.0 to 0.1), to prevent loss of useful information during training while avoiding overfitting.
- **Hidden Dimensions:** `(200, 200)`, as it seems to provide adequate complexity without leading to overfitting.
- **Optimizer:** Adam, due to its efficient handling of sparse and noisy data, leading to better and more stable convergence.

---

### Python Files Explanation

#### 1. **`gram.py`**:
Here's a thorough explanation of the code in the `gram.py` file in markdown format:

### 5-Gram Neural Language Model with Word2Vec and PyTorch

### Step 1: Text Preprocessing, Vocabulary, N-Grams

```python
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
```

- The script reads a text file `Auguste_Maquet.txt`, converts it to lowercase, and removes any characters that are not letters or whitespace using regular expressions.
- The text is then split into individual words.

- **`word_count`** counts the frequency of each word.
- **`vocab`** is a dictionary that assigns a unique index to each word in the text. Words that occur at least once are included, and the special token `<UNK>` is used for unknown words.

- The `tokenize_word` function returns the corresponding index for a given word, or the `<UNK>` token if the word is not found in the vocabulary.
- **`tokenized_text`** is a list of tokenized words from the input text.

- The **`generate_ngrams`** function creates 5-grams from the tokenized text. Each entry in the resulting list contains a tuple of (context, target word), where the context is a sequence of 5 words, and the target word is the next word in the text.

```python
train_data, test_data = train_test_split(ngrams, test_size=0.1, random_state=42)
```

- The n-grams are split into training and testing sets, with 10% of the data reserved for testing.

### Step 2: Define the Neural Language Model

```python
class NeuralLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim1=300, hidden_dim2=300):
        super(NeuralLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 5, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        embedded = self.embedding(x).view(x.size(0), -1)
        hidden1 = F.relu(self.fc1(embedded))
        hidden2 = F.relu(self.fc2(hidden1))
        output = self.fc3(hidden2)
        return output
```

- The `NeuralLanguageModel` class is a feedforward neural network:
    - **Embedding layer**: Converts words into Word2Vec embeddings.
    - **Fully connected layers (fc1, fc2, fc3)**: The 5-grams are passed through two hidden layers and finally projected onto the vocabulary size.
    - **Softmax**: Applied to the final layer's output to produce a probability distribution over the vocabulary.

### Step 3: Initialize the Model and Load Word2Vec Embeddings

```python
vocab_size = len(vocab) + 1
embedding_dim = 100
model = NeuralLanguageModel(vocab_size, embedding_dim).to('mps')
```

- The vocabulary size includes the `<UNK>` token, and the embedding dimension is set to 100. 

```python
w2v_model = Word2Vec([words], vector_size=embedding_dim, min_count=1)
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in vocab.items():
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]
model.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
```

- A Word2Vec model is trained on the words, and its embeddings are transferred to the neural model's embedding layer.

### Step 4: NGramDataset Class

```python
class NGramDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx][0]), torch.tensor(self.data[idx][1])
```

- The `NGramDataset` class wraps the training data in a PyTorch Dataset, which allows for easy batching and shuffling using DataLoader.

### Step 5: Training Loop

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def calculate_perplexity(loss):
    return torch.exp(loss)

epochs = 10
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
```

- The model is trained for 10 epochs, using cross-entropy loss and the Adam optimizer. The model predicts the next word based on the 5-gram context. After each epoch, the average loss and perplexity are calculated and displayed.

```python
model_path = "5gram_language_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
```

- After training, the model's weights are saved to a file named `5gram_language_model.pth`.


#### 2. **`hyp_gram.py`**:
This file is responsible for hyperparameter tuning and managing different configurations.
- **Hyperparameter Sweeps**: Adjusts dropout rates, hidden layer dimensions, and optimizers to identify the best-performing configuration.
- **Logging and Plotting**: Saves results like perplexity and generates plots for analysis. It also exports the results to a CSV for further review.

#### 3. **`run_gram.py`**:
This file acts as the main point to run the trained models.
- **Model Initialization**: Instantiates the language model using parameters like vocabulary size and embedding dimension.
- **Results**: Handles the final saving of models and predicts results from the trained model.
