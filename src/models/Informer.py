class IntentionBlock(nn.Module):
    def __init__(self, embed_dim, ff_dim, alpha=0.1, dropout=0.05, normalize_keys=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.alpha = alpha
        self.normalize_keys = normalize_keys
        
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        
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
        K = self.W_k(x)
        Q = self.W_q(x)
        V = self.W_v(x)
        
        # L2 normalize keys for stability (from paper)
        if self.normalize_keys:
            K = nn.functional.normalize(K, p=2, dim=-1)
        
        K_T = K.transpose(1, 2)
        KTK = torch.bmm(K_T, K)
        
        # Regularization
        reg = self.alpha * torch.eye(self.embed_dim, device=x.device).unsqueeze(0)
        KTK_reg = KTK + reg
        
        KTV = torch.bmm(K_T, V)
        
        # Solve with better numerical stability
        try:
            solved = torch.linalg.solve(KTK_reg, KTV)
        except:
            # Fallback to pseudo-inverse if singular
            solved = torch.linalg.lstsq(KTK_reg, KTV).solution
        
        intention_output = torch.bmm(Q, solved)
        
        x = self.layernorm1(x + self.dropout1(intention_output))
        ffn_output = self.ffn(x)
        x = self.layernorm2(x + self.dropout2(ffn_output))
        
        return x

class TabularInformer(nn.Module):
    def __init__(self, n_features, embed_dim=64, ff_dim=128, num_blocks=2, alpha=0.01, dropout=0.05):
        super().__init__()
        
        # Feature embedding
        self.feature_embedding = nn.Linear(1, embed_dim)
        
        # Positional embedding
        self.pos_embedding = nn.Embedding(n_features, embed_dim)
        
        # Stack of Intention blocks
        self.blocks = nn.ModuleList([
            IntentionBlock(embed_dim, ff_dim, alpha, dropout)
            for _ in range(num_blocks)
        ])
        
        # Output head
        self.output = nn.Linear(embed_dim, 1)

    def forward(self, x):
        batch_size, n_features = x.shape
        
        # Embed features
        x = x.unsqueeze(-1)
        x = self.feature_embedding(x)
        
        # Add positional embeddings
        positions = torch.arange(n_features, device=x.device)
        pos_emb = self.pos_embedding(positions)
        x = x + pos_emb.unsqueeze(0)
        
        # Pass through Intention blocks
        for block in self.blocks:
            x = block(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        return self.output(x).squeeze(-1)
