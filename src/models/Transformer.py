import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x):
        # Multi-head attention with residual
        attn_output, _ = self.attention(x, x, x)
        x = self.layernorm1(x + self.dropout1(attn_output))
        
        # Feed-forward with residual
        ffn_output = self.ffn(x)
        x = self.layernorm2(x + self.dropout2(ffn_output))
        return x

class TabularTransformer(nn.Module):
    def __init__(self, n_features, embed_dim=64, num_heads=4, ff_dim=128, num_blocks=2, dropout=0.1):
        super().__init__()
        
        # Feature embedding - each feature gets embedded
        self.feature_embedding = nn.Linear(1, embed_dim)
        
        # Positional embedding for feature positions
        self.pos_embedding = nn.Embedding(n_features, embed_dim)
        
        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_blocks)
        ])
        
        # Output head
        self.output = nn.Linear(embed_dim, 1)
        
    def forward(self, x):
        # x shape: (batch, n_features)
        batch_size, n_features = x.shape
        
        # Reshape to (batch, n_features, 1) for embedding
        x = x.unsqueeze(-1)
        
        # Embed each feature
        x = self.feature_embedding(x)  # (batch, n_features, embed_dim)
        
        # Add positional embeddings
        positions = torch.arange(n_features, device=x.device)
        pos_emb = self.pos_embedding(positions)  # (n_features, embed_dim)
        x = x + pos_emb.unsqueeze(0)  # broadcast across batch
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch, embed_dim)
        
        # Final prediction
        return self.output(x).squeeze(-1)
