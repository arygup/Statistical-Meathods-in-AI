import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from gensim.models import Word2Vec
import re
from collections import Counter
import os

def tok_word(wd):
    return voc.get(wd, voc['<UNK>'])
def pad_seqs(seqs, mx_len, pad_val=0):
    pad_seqs = []
    for sq in seqs:
        if len(sq) < mx_len:
            pad_sq = sq + [pad_val] * (mx_len - len(sq))
        else:
            pad_sq = sq[:mx_len]
        pad_seqs.append(pad_sq)
    return pad_seqs

dev = torch.device('mps' if torch.backends.mps.is_built() else 'cpu')
with open('Auguste_Maquet.txt', 'r') as f:
    txt = f.read().lower()

sents = re.split(r'[.!?]', txt)
sents = [re.sub(r'[^a-z\s]', '', sent).split() for sent in sents if sent.strip()]
filtered_sents = [sent for sent in sents if len(sent) <= 20]
w2v_model = Word2Vec(filtered_sents, vector_size=128, window=5, min_count=1, sg=1)
wds = [wd for sent in filtered_sents for wd in sent]
wd_count = Counter(wds)
voc = {wd: i+1 for i, (wd, count) in enumerate(wd_count.items()) if count >= 1}
voc['<UNK>'] = len(voc) + 1
voc['<PAD>'] = 0
tok_sents = [[tok_word(wd) for wd in sent] for sent in filtered_sents]
mx_len = 20
padded_X = pad_seqs([sent[:-1] for sent in tok_sents if len(sent) > 1], mx_len)
padded_y = pad_seqs([sent[1:] for sent in tok_sents if len(sent) > 1], mx_len)
split_rt = 0.8
split_idx = int(len(padded_X) * split_rt)

X_tr = torch.tensor(padded_X[:split_idx], dtype=torch.long)
y_tr = torch.tensor(padded_y[:split_idx], dtype=torch.long)
X_ts = torch.tensor(padded_X[split_idx:], dtype=torch.long)
y_ts = torch.tensor(padded_y[split_idx:], dtype=torch.long)

class DecModel(nn.Module):
    def __init__(self, voc_sz, emb_dim, nheads, n_layers, ctxt_len):
        super(DecModel, self).__init__()
        self.emb = nn.Embedding(voc_sz, emb_dim)
        self.pos_enc = nn.Parameter(torch.zeros(1, ctxt_len, emb_dim))
        self.dec_layer = nn.TransformerDecoderLayer(d_model=emb_dim, nhead=nheads)
        self.trans_dec = nn.TransformerDecoder(self.dec_layer, num_layers=n_layers)
        self.fc_out = nn.Linear(emb_dim, voc_sz)

    def forward(self, inp, pad_msk=None):
        inp = self.emb(inp) + self.pos_enc
        sq_len = inp.size(1)
        msk = self._gen_sq_mask(sq_len).to(inp.device)
        inp = self.trans_dec(inp.transpose(0, 1), inp.transpose(0, 1), tgt_mask=msk, tgt_key_padding_mask=pad_msk)
        inp = self.fc_out(inp.transpose(0, 1))
        return inp

    def _gen_sq_mask(self, sq_len):
        msk = (torch.triu(torch.ones(sq_len, sq_len)) == 1).transpose(0, 1)
        msk = msk.float().masked_fill(msk == 0, float('-inf')).masked_fill(msk == 1, float(0.0))
        return msk

emb_dim = 128
nheads = 4
n_layers = 4
ctxt_len = mx_len
voc_sz = len(voc)
tr_ds = TensorDataset(X_tr, y_tr)
ts_ds = TensorDataset(X_ts, y_ts)
b_sz = 64
tr_ld = DataLoader(tr_ds, batch_size=b_sz, shuffle=True)
ts_ld = DataLoader(ts_ds, batch_size=b_sz, shuffle=False)
n_tr_batches = len(tr_ld)
n_ts_batches = len(ts_ld)
loss_fn = nn.CrossEntropyLoss(ignore_index=voc['<PAD>'])
mdl = DecModel(voc_sz, emb_dim, nheads, n_layers, ctxt_len).to(dev)
optim = torch.optim.Adam(mdl.parameters(), lr=0.01)

def calculate_epoch_perplexity(mdl, ld, loss_fn):
    mdl.eval()
    sentence_perplexities = []
    with torch.no_grad():
        for bx, by in ld:
            bx = bx.to(dev)
            by = by.to(dev)
            pad_msk = (bx == voc['<PAD>'])
            outs = mdl(bx, pad_msk)
            outs = outs.view(bx.size(0), -1, outs.size(-1))  
            by = by.view(bx.size(0), -1)  
            for i in range(bx.size(0)):  
                sentence_loss = loss_fn(outs[i], by[i])  
                sentence_perplexity = torch.exp(sentence_loss).item()  
                sentence_perplexities.append(sentence_perplexity)  
    avg_perplexity = sum(sentence_perplexities) / len(sentence_perplexities)
    return avg_perplexity


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
            outs = outs.view(-1, outs.size(-1))
            by = by.view(-1)
            loss = loss_fn(outs, by)
            loss.backward()
            optim.step()
            run_loss += loss.item()
        train_perplexity = calculate_epoch_perplexity(mdl, tr_ld, loss_fn)
        print(f'Epoch [{ep+1}/{n_epochs}], Train Loss: {run_loss / len(tr_ld)}, Train Perplexity: {train_perplexity:.4f}')
        torch.save(mdl.state_dict(), f'{ckpt_path}/mdl_ep_{ep+1}.pth')
    return 

n_epochs = 15
ckpt_dir = 'mdl_ckpts'
train_model(mdl, tr_ld, loss_fn, optim, n_epochs, ckpt_path=ckpt_dir)


def load_mdl(mdl, ckpt_path):
    mdl.load_state_dict(torch.load(ckpt_path))
    mdl.to(dev)
    mdl.eval()
    return mdl

ckpt_path = os.path.join(ckpt_dir, f'mdl_ep_{n_epochs}.pth')
mdl = load_mdl(mdl, ckpt_path)

def calc_perplexity_and_save_corrected(mdl, ts_ld, loss_fn, vocab, output_file):
    mdl.eval()
    with torch.no_grad(), open(output_file, 'w') as f_out:
        for bx, by in ts_ld:
            bx = bx.to(dev)
            by = by.to(dev)
            pad_msk = (bx == voc['<PAD>'])
            outs = mdl(bx, pad_msk)
            outs = outs.view(-1, outs.size(-1))
            by = by.view(-1)
            for i in range(bx.size(0)):
                sentence_loss = loss_fn(outs.view(bx.size(0), -1, outs.size(-1))[i], by.view(bx.size(0), -1)[i])
                sentence_perplexity = torch.exp(sentence_loss).item()
                sentence = ' '.join([list(vocab.keys())[list(vocab.values()).index(token.item())] for token in bx[i] if token.item() != voc['<PAD>']])
                f_out.write(f"{sentence}\t{sentence_perplexity:.4f}\n")
    return 

output_file = "perplexity_scores.txt"
calc_perplexity_and_save_corrected(mdl, ts_ld, loss_fn, voc, output_file)
print(f'Perplexity on the test set')