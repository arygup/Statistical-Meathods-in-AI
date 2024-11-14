### Building an LSTM-based Language Model with PyTorch


---

#### 1. **Data Preprocessing**

The first part of the code reads and preprocesses text data:

- **Reading Text**:  
  The text is read from the file `Auguste_Maquet.txt`, converted to lowercase.

  ```python
  with open('Auguste_Maquet.txt', 'r') as f:
      text = f.read().lower()
  ```

- **Sentence Tokenization**:  
  The text is split into sentences based on punctuation, and non-alphabetical characters are removed. Each sentence is further split into words.

  ```python
  sentences = re.split(r'[.!?]', text)
  sentences = [re.sub(r'[^a-z\s]', '', sentence).split() for sentence in sentences if sentence.strip()]
  ```

- **Building Vocabulary**:  
  A vocabulary is created from the words in the sentences, and each word is assigned a unique index. An `<UNK>` token is used for unknown words.

  ```python
  words = [word for sentence in sentences for word in sentence]
  word_count = Counter(words)
  vocab = {word: i+1 for i, (word, count) in enumerate(word_count.items()) if count >= 1}
  vocab['<UNK>'] = len(vocab) + 1
  ```

- **Tokenizing Sentences**:  
  Each word in the sentences is converted to its corresponding index from the vocabulary.

  ```python
  tokenized_sentences = [[tokenize_word(word) for word in sentence] for sentence in sentences]
  ```

- **Filtering Sentences**:  
  Only sentences with a length of 20 words or less are kept.

  ```python
  tokenized_sentences = [sentence for sentence in tokenized_sentences if len(sentence) <= 20]
  ```

#### 2. **Dataset and DataLoader**

A custom `Dataset` class, `SentenceDataset`, is created to handle the sentences and corresponding target words for training. The targets are simply the next word in the sentence.

- **Dataset**:  
  For each sentence, the input is the sequence of words except the last word, and the target is the sequence shifted by one word.

  ```python
  class SentenceDataset(Dataset):
      def __init__(self, data):
          self.data = data
      
      def __len__(self):
          return len(self.data)
      
      def __getitem__(self, idx):
          sentence = torch.tensor(self.data[idx][:-1], dtype=torch.long)
          target = torch.tensor(self.data[idx][1:], dtype=torch.long)
          return sentence, target
  ```

- **Collate Function**:  
  A custom `collate_fn` is defined to pad sequences in a batch, ensuring that all sequences in a batch have the same length.

  ```python
  def collate_fn(batch):
      sentences, targets = zip(*batch)
      sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=0)
      targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
      return sentences_padded, targets_padded
  ```

#### 3. **LSTM-based Neural Language Model**

The `LSTMLanguageModel` class defines the architecture of the neural network:

- **Embedding Layer**:  
  Converts word indices into dense vector representations.

  - `nn.Embedding(vocab_size, embedding_dim)`

- **LSTM Layer**:  
  Processes the input sequence using an LSTM (Long Short-Term Memory) layer.

  - `nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)`

- **Fully Connected Layer**:  
  Transforms the LSTM output to match the size of the vocabulary.

  - `nn.Linear(hidden_dim, vocab_size)`

  ```python
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
  ```

#### 4. **Word2Vec Embeddings**

Word2Vec embeddings are used to initialize the weights of the embedding layer. The pretrained embeddings are loaded, and the embedding matrix is copied into the model.

```python
w2v_model = Word2Vec([words], vector_size=embedding_dim, min_count=1)
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in vocab.items():
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]
model.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
```

#### 5. **Training the Model**

The model is trained using cross-entropy loss (`nn.CrossEntropyLoss`) and the Adam optimizer (`optim.Adam`):

- **Training Loop**:  
  For each epoch, the model processes batches of sentences and adjusts weights using backpropagation.

  ```python
  epochs = 10
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
  ```

- **Perplexity**:  
  The perplexity metric is used to evaluate the model's performance. It is a measure of how well the model predicts the next word in a sequence.

  ```python
  def calculate_perplexity(loss):
      return torch.exp(loss)
  ```

#### 6. **Saving the Model**

At the end of each epoch, the model’s state is saved to a file:

```python
model_path = "lstm_language_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
```

---

This LSTM-based language model processes sentences from text, predicts the next word, and adjusts its internal parameters to improve future predictions. The model is trained word-by-word, using word embeddings, LSTM layers for sequential processing, and padding to handle varying sentence lengths.