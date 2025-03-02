import torch
import torch.nn as nn
import math
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from collections import Counter
from encoder import Encoder
from decoder import Decoder
import pickle
from utils import (
    tokenize_english, tokenize_french, read_and_tokenize, build_vocabulary,
    convert_tokens_to_indices, pad_sentences_to_length
)
nltk.download('punkt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# device = torch.device('mps')
# print(f"Using device: {device}")

# max_allowed_length = 3


max_allowed_length = 100


file_paths = {
    'test_en': 'test.en',
    'test_fr': 'test.fr'
}

# Load test data
test_en_sentences = read_and_tokenize(file_paths['test_en'], tokenize_english)
test_fr_sentences = read_and_tokenize(file_paths['test_fr'], tokenize_french)

# Filter sentences based on length
filtered_test_en_sentences = [s for s in test_en_sentences if len(s) <= max_allowed_length]
filtered_test_fr_sentences = [
    test_fr_sentences[i] for i in range(len(test_en_sentences))
    if len(test_en_sentences[i]) <= max_allowed_length
]

assert len(filtered_test_en_sentences) == len(filtered_test_fr_sentences), "Mismatch in sentence counts."

with open('en_vocab.pkl', 'rb') as f:
    en_vocab = pickle.load(f)
with open('fr_vocab.pkl', 'rb') as f:
    fr_vocab = pickle.load(f)

# Define the Transformer model (same as in train.py)
class Transformer(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size, context_dim, num_heads, num_layers, compression_factor, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.enc_emb = nn.Embedding(source_vocab_size, context_dim)
        self.dec_emb = nn.Embedding(target_vocab_size, context_dim)

        self.pe = self.create_positional_encoding(max_seq_length, context_dim).to(device)
        self.dropout = nn.Dropout(dropout)

        self.encoders = nn.ModuleList([
            Encoder(context_dim, num_heads, compression_factor, dropout) for _ in range(num_layers)
        ])
        self.decoders = nn.ModuleList([
            Decoder(context_dim, num_heads, compression_factor, dropout) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(context_dim, target_vocab_size)

    def create_positional_encoding(self, max_seq_length, context_dim):
        max_seq_length += 400
        pe = torch.zeros(max_seq_length, context_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, context_dim, 2) * (-math.log(10000.0) / context_dim))

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        return pe

    def add_positional_encoding(self, x):
        seq_len = x.size(1)
        pe = self.pe[:seq_len, :].unsqueeze(0).to(x.device)
        return x + pe

    def forward(self, source, target):
        source_mask = (source != en_vocab['<pad>']).unsqueeze(1).unsqueeze(2)
        target_mask = (target != fr_vocab['<pad>']).unsqueeze(1).unsqueeze(2)
        seq_len = target.size(1)
        nopeak_mask = torch.tril(torch.ones((1, 1, seq_len, seq_len), device=target.device)).bool()
        target_mask = target_mask & nopeak_mask

        # Embeddings
        source_embedded = self.dropout(self.add_positional_encoding(self.enc_emb(source)))
        target_embedded = self.dropout(self.add_positional_encoding(self.dec_emb(target)))

        # Encoder
        enc_output = source_embedded
        for encoder in self.encoders:
            enc_output = encoder(enc_output, source_mask)

        # Decoder
        dec_output = target_embedded
        for decoder in self.decoders:
            dec_output = decoder(dec_output, enc_output, source_mask, target_mask)

        output = self.fc_out(dec_output)
        return output

# Load the saved model
model = Transformer(
    source_vocab_size=len(en_vocab),
    target_vocab_size=len(fr_vocab),
    context_dim=512,
    num_heads=4,
    num_layers=4,
    compression_factor=4,
    max_seq_length=max_allowed_length,
    dropout=0.3
).to(device)

model.load_state_dict(torch.load('transformer.pt', map_location=device))
model.eval()

# Evaluation functions
def translate_and_evaluate_bleu(model, en_sentence, fr_sentence, en_vocab, fr_vocab, max_length=int(max_allowed_length * 1.3)):
    tokens = en_sentence
    token_indices = [en_vocab.get(token, en_vocab['<unk>']) for token in tokens]
    token_indices = [en_vocab['<sos>']] + token_indices + [en_vocab['<eos>']]
    token_indices += [en_vocab['<pad>']] * (max_length - len(token_indices))
    source_tensor = torch.tensor(token_indices).unsqueeze(0).to(device)
    target_indices = [fr_vocab['<sos>']]
    output_tokens = []
    for _ in range(max_length):
        target_tensor = torch.tensor(target_indices).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(source_tensor, target_tensor)
        next_token_idx = output.argmax(-1)[:, -1].item()
        target_indices.append(next_token_idx)
        output_token = [k for k, v in fr_vocab.items() if v == next_token_idx][0]
        if output_token == '<eos>':
            break
        output_tokens.append(output_token)
    reference_tokens = [word for word in fr_sentence if word not in ['<sos>', '<eos>', '<pad>']]
    smoothing = SmoothingFunction().method1
    bleu_score = sentence_bleu([reference_tokens], output_tokens, smoothing_function=smoothing)
    return bleu_score, output_tokens, reference_tokens

# Evaluate on test set
def evaluate_test_set(model, test_en_sentences, test_fr_sentences, en_vocab, fr_vocab):
    total_bleu_score = 0.0
    for en_sentence, fr_sentence in zip(test_en_sentences, test_fr_sentences):
        bleu_score, pred_tokens, ref_tokens = translate_and_evaluate_bleu(
            model, en_sentence, fr_sentence, en_vocab, fr_vocab
        )
        total_bleu_score += bleu_score
        print(f"English: {' '.join(en_sentence)}")
        print(f"Reference French: {' '.join(ref_tokens)}")
        print(f"Predicted French: {' '.join(pred_tokens)}")
        print(f"BLEU Score: {bleu_score:.4f}\n")
    avg_bleu_score = total_bleu_score / len(test_en_sentences)
    print(f"Average BLEU Score on Test Set: {avg_bleu_score:.4f}")

# Run evaluation
evaluate_test_set(model, filtered_test_en_sentences, filtered_test_fr_sentences, en_vocab, fr_vocab)
