# Transformer-based Language Model 

### 1. **Text Preprocessing and Tokenization**
The code starts by loading and processing the text file:

```python
with open('Auguste_Maquet.txt', 'r') as f:
    txt = f.read().lower()

sents = re.split(r'[.!?]', txt)
sents = [re.sub(r'[^a-z\s]', '', sent).split() for sent in sents if sent.strip()]
filtered_sents = [sent for sent in sents if len(sent) <= 20]
```
- Text is split into sentences and tokenized. 
- Sentences longer than 20 words are filtered out.

Next, the vocabulary is created using the `Word2Vec` model:

```python
w2v_model = Word2Vec(filtered_sents, vector_size=128, window=5, min_count=1, sg=1)
```
- The model learns word embeddings with a size of 128 for all tokens, using a skip-gram approach (`sg=1`).

---

### 2. **Padding and Dataset Preparation**
The sentences are padded to a maximum length of 20:

```python
padded_X = pad_seqs([sent[:-1] for sent in tok_sents if len(sent) > 1], mx_len)
padded_y = pad_seqs([sent[1:] for sent in tok_sents if len(sent) > 1], mx_len)
```
- `padded_X`: The input sequences (all tokens except the last).
- `padded_y`: The target sequences (all tokens except the first).

This data is split into training and test sets (80/20 split):

```python
split_rt = 0.8
split_idx = int(len(padded_X) * split_rt)

X_tr = torch.tensor(padded_X[:split_idx], dtype=torch.long)
y_tr = torch.tensor(padded_y[:split_idx], dtype=torch.long)
```

---

### 3. **Model Architecture: Transformer Decoder**
The model is built using the `nn.TransformerDecoder` module:

```python
class DecModel(nn.Module):
    def __init__(self, voc_sz, emb_dim, nheads, n_layers, ctxt_len):
        super(DecModel, self).__init__()
        self.emb = nn.Embedding(voc_sz, emb_dim)
        self.pos_enc = nn.Parameter(torch.zeros(1, ctxt_len, emb_dim))
        self.dec_layer = nn.TransformerDecoderLayer(d_model=emb_dim, nhead=nheads)
        self.trans_dec = nn.TransformerDecoder(self.dec_layer, num_layers=n_layers)
        self.fc_out = nn.Linear(emb_dim, voc_sz)
```
- **Embedding layer**: Converts token indices into embeddings.
- **Positional Encoding**: Adds positional information to the embeddings.
- **Transformer Decoder**: Multiple decoder layers stacked with multi-head attention.
- **Output Layer**: A linear layer maps the decoder's output to vocabulary size.

The model generates a square mask for attention during decoding:

```python
def _gen_sq_mask(self, sq_len):
    msk = (torch.triu(torch.ones(sq_len, sq_len)) == 1).transpose(0, 1)
    msk = msk.float().masked_fill(msk == 0, float('-inf')).masked_fill(msk == 1, float(0.0))
    return msk
```

---

### 4. **Training the Model**
The training loop uses `CrossEntropyLoss` and an Adam optimizer:

```python
def train_model(mdl, tr_ld, loss_fn, optim, n_epochs, ckpt_path=None):
    for ep in range(n_epochs):
        mdl.train()
        run_loss = 0.0
        for bx, by in tr_ld:
            optim.zero_grad()
            bx = bx.to(dev)
            by = by.to(dev)
            pad_msk = (bx == voc['<PAD>'])
            outs = mdl(bx, pad_msk)
            loss = loss_fn(outs.view(-1, outs.size(-1)), by.view(-1))
            loss.backward()
            optim.step()
            run_loss += loss.item()
        # Save the model
        torch.save(mdl.state_dict(), f'{ckpt_path}/mdl_ep_{ep+1}.pth')
```
- The model is trained over multiple epochs, with loss being accumulated and backpropagated.

---

### 5. **Perplexity Calculation**
The model is evaluated using perplexity, which is a measure of how well a probability distribution or probability model predicts a sample:

```python
def calculate_epoch_perplexity(mdl, ld, loss_fn):
    mdl.eval()
    sentence_perplexities = []
    with torch.no_grad():
        for bx, by in ld:
            bx = bx.to(dev)
            by = by.to(dev)
            outs = mdl(bx)
            loss = loss_fn(outs.view(-1, outs.size(-1)), by.view(-1))
            sentence_perplexity = torch.exp(loss).item()
            sentence_perplexities.append(sentence_perplexity)
    return sum(sentence_perplexities) / len(sentence_perplexities)
```
- **Perplexity**: Calculated as the exponential of the cross-entropy loss.

---

### 6. **Saving Predictions and Perplexity**
After training, the model is used to compute and save perplexities for the test dataset:

```python
def calc_perplexity_and_save_corrected(mdl, ts_ld, loss_fn, vocab, output_file):
    with open(output_file, 'w') as f_out:
        for bx, by in ts_ld:
            sentence_perplexity = calculate_epoch_perplexity(mdl, ts_ld, loss_fn)
            f_out.write(f'{sentence}\t{sentence_perplexity}\n')
```
- The predictions and their perplexities are saved to an output file.

---