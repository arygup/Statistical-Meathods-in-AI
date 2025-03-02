import torch.nn as nn
from utils import MultiHeadAttention, FeedForwardNN

class Decoder(nn.Module):
    def __init__(self, context_dim, num_heads, compression_factor, dropout):
        super(Decoder, self).__init__()
        self.drp = nn.Dropout(dropout)
        self.attn = MultiHeadAttention(context_dim, num_heads)
        self.n1 = nn.LayerNorm(context_dim)
        self.cross_attn = MultiHeadAttention(context_dim, num_heads)
        self.n2 = nn.LayerNorm(context_dim)
        self.ffn = FeedForwardNN(context_dim, compression_factor)
        self.n3 = nn.LayerNorm(context_dim)

    def forward(self, x, encoding, source_mask, target_mask):
        # Masked Self-Attention
        attn = self.attn(x, x, x, target_mask)
        x = self.n1(x + self.drp(attn))
        # Cross-Attention with Encoder Output
        cross = self.cross_attn(x, encoding, encoding, source_mask)
        x = self.n2(x + self.drp(cross))
        # Feed Forward Network
        ffn = self.ffn(x)
        x = self.n3(x + self.drp(ffn))
        return x
