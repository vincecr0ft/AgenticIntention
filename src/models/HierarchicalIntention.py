class HierarchicalIntention(nn.Module):
    def __init__(self, n_features, embed_dim=64, ff_dim=128):
        super().__init__()
        
        # Feature embedding (same as original)
        self.feature_embedding = nn.Linear(1, embed_dim)
        self.pos_embedding = nn.Embedding(n_features, embed_dim)
        
        # THREE different intention blocks with different alphas
        self.coarse_intention = IntentionBlock(embed_dim, ff_dim, alpha=1.0)    # High regularization
        self.medium_intention = IntentionBlock(embed_dim, ff_dim, alpha=0.1)    # Medium
        self.fine_intention = IntentionBlock(embed_dim, ff_dim, alpha=0.01)    # Low regularization
        
        # Gating network to blend the three
        self.gate_network = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # 3 weights for 3 intention blocks
        )
        
        # Output head
        self.output = nn.Linear(embed_dim, 1)
        
    def forward(self, x):
        # Standard embedding
        x = x.unsqueeze(-1)
        x_embedded = self.feature_embedding(x)
        positions = torch.arange(x.shape[1], device=x.device)
        x_embedded = x_embedded + self.pos_embedding(positions).unsqueeze(0)
        
        # Get three different representations
        coarse_out = self.coarse_intention(x_embedded)
        medium_out = self.medium_intention(x_embedded)
        fine_out = self.fine_intention(x_embedded)
        
        # Learn how to weight them based on global pool of input
        gate_input = x_embedded.mean(dim=1)  # (batch, embed_dim)
        gate_weights = torch.softmax(self.gate_network(gate_input), dim=-1)  # (batch, 3)
        
        # Weighted combination
        combined = (gate_weights[:, 0].unsqueeze(1).unsqueeze(2) * coarse_out +
                   gate_weights[:, 1].unsqueeze(1).unsqueeze(2) * medium_out +
                   gate_weights[:, 2].unsqueeze(1).unsqueeze(2) * fine_out)
        
        # Global average pooling and output
        x = combined.mean(dim=1)
        return self.output(x).squeeze(-1)
