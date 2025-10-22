class TabularIntention(nn.Module):
    def __init__(self, n_features, embed_dim=128, ff_dim=256, alpha=0.01, dropout=0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Feature-specific learnable embeddings (identity of each feature)
        self.feature_keys = nn.Parameter(torch.randn(n_features, embed_dim))
        self.feature_values = nn.Parameter(torch.randn(n_features, embed_dim))
        self.feature_queries = nn.Parameter(torch.randn(n_features, embed_dim))
        
        # Value encoder - how we incorporate the actual data values
        self.value_encoder = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # The core intention mechanism
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # Crucial: learn feature interactions explicitly
        self.interaction_layer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, 1)
        )
        
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, n_features = x.shape
        
        # Step 1: Encode values and combine with feature identity
        # This respects that "age=25" means something different than "income=25"
        value_embeds = self.value_encoder(x.unsqueeze(-1))  # (batch, n_features, embed_dim)
        
        # Step 2: Create feature-aware K, V, Q
        # Keys: what information does each feature provide?
        K = self.W_k(self.feature_keys.unsqueeze(0) + value_embeds)
        
        # Values: what should be propagated from each feature?
        V = self.W_v(self.feature_values.unsqueeze(0) + value_embeds)
        
        # Queries: what does each feature need to know?
        Q = self.W_q(self.feature_queries.unsqueeze(0) + value_embeds)
        
        # Step 3: The intention solve - but feature-wise
        # Transpose to work on features not samples
        K_T = K.transpose(1, 2)  # (batch, embed_dim, n_features)
        
        # Compute the feature relationship matrix
        KTK = torch.bmm(K_T, K)  # (batch, embed_dim, embed_dim)
        
        # Regularization for numerical stability
        reg = self.alpha * torch.eye(self.embed_dim, device=x.device).unsqueeze(0)
        KTK_reg = KTK + reg
        
        # Project values into the relationship space
        KTV = torch.bmm(K_T, V)  # (batch, embed_dim, embed_dim)
        
        # Solve for the optimal feature transformation
        try:
            beta = torch.linalg.solve(KTK_reg, KTV)
        except:
            beta = torch.linalg.lstsq(KTK_reg, KTV).solution
            
        # Apply the learned transformation
        intention_output = torch.bmm(Q, beta)  # (batch, n_features, embed_dim)
        
        # Step 4: Learn pairwise feature interactions explicitly
        # This is crucial for credit scoring where interactions matter
        feature_interactions = []
        for i in range(n_features):
            for j in range(i+1, n_features):
                interaction = self.interaction_layer(
                    torch.cat([intention_output[:, i], intention_output[:, j]], dim=-1)
                )
                feature_interactions.append(interaction)
        
        # Combine all interactions
        if feature_interactions:
            interactions = torch.stack(feature_interactions, dim=1).mean(dim=1)
        else:
            interactions = 0
        
        # Step 5: Global aggregation with interaction term
        global_features = intention_output.mean(dim=1)  # (batch, embed_dim)
        
        # Add explicit interaction signal
        if isinstance(interactions, torch.Tensor):
            combined = global_features + 0.1 * interactions  # Weight interactions less
        else:
            combined = global_features
            
        # Step 6: Final prediction
        output = self.output_proj(combined)
        
        return output.squeeze(-1)
