import torch.nn as nn
from utils import MultiHeadAttention, FeedForwardNN

class Encoder(nn.Module):
    def __init__(self, context_dim, num_heads, compression_factor, dropout):
        super(Encoder, self).__init__()
        self.drp = nn.Dropout(dropout)
        self.attn = MultiHeadAttention(context_dim, num_heads)
        self.n1 = nn.LayerNorm(context_dim)
        self.ffn = FeedForwardNN(context_dim, compression_factor)
        self.n2 = nn.LayerNorm(context_dim)

    def forward(self, x, mask=None):
        # Attention -> Layer Norm -> Feed Forward NN -> Add & Norm
        attn = self.attn(x, x, x, mask)
        x = self.n1(x + self.drp(attn))
        ffn = self.ffn(x)
        x = self.n2(x + self.drp(ffn))
        return x
