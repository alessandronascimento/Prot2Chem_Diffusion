import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class AdaLN(nn.Module):
    def __init__(self, hidden_dim, cond_dim):
        """
        hidden_dim: The size of your latent vector (e.g., 256)
        cond_dim: The size of your global condition vector (Time + Affinity)
        """
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.linear = nn.Linear(cond_dim, 2 * hidden_dim)
        
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, c_global):
        # x shape: [Batch, Seq_Len, Hidden_Dim]
        # c_global shape: [Batch, Cond_Dim]
        
        scale_shift = self.linear(c_global).unsqueeze(1)
        
        # Split into gamma (scale) and beta (shift)
        gamma, beta = scale_shift.chunk(2, dim=2)
        
        # Apply normalization and steer!
        return self.norm(x) * (1 + gamma) + beta

class CrossAttention(nn.Module):
    def __init__(self, hidden_dim, context_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Projection layers (Mapping 1280 context to 256)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(context_dim, hidden_dim)
        self.v_proj = nn.Linear(context_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, context):
        # x: [Batch, 1, 256] (The Latent)
        # context: [Batch, 1, 1280] (The Protein)
        B, seq_len_q, _ = x.size()
        _, seq_len_k, _ = context.size()

        Q = self.q_proj(x)       # [B, 1, 256]
        K = self.k_proj(context) # [B, 1, 256]
        V = self.v_proj(context) # [B, 1, 256]
        
        Q = Q.view(B, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_output = F.scaled_dot_product_attention(Q, K, V)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, seq_len_q, -1)
        
        return self.out_proj(attn_output)

class ConditionalDiTBlock(nn.Module):
    def __init__(self, hidden_dim=256, cond_dim=256, context_dim=1280, num_heads=8):
        super().__init__()
        
        # Self-Attention + AdaLN
        self.norm1 = AdaLN(hidden_dim, cond_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        # Cross-Attention + AdaLN
        self.norm2 = AdaLN(hidden_dim, cond_dim)
        self.cross_attn = CrossAttention(hidden_dim, context_dim, num_heads)
        
        # Feed-Forward Network + AdaLN
        self.norm3 = AdaLN(hidden_dim, cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, x, c_global, c_spatial):
        """
        x: Noisy Latent [Batch, 1, 256]
        c_global: Time + Affinity Embedding [Batch, 256]
        c_spatial: Protein Embedding [Batch, 1, 1280]
        """
        nx = self.norm1(x, c_global)
        attn_out, _ = self.self_attn(nx, nx, nx)
        x = x + attn_out
        
        # Cross-Attention to Protein
        nx = self.norm2(x, c_global)
        x = x + self.cross_attn(nx, c_spatial)
        
        # MLP
        nx = self.norm3(x, c_global)
        x = x + self.mlp(nx)
        
        return x

class TimestepEmbedder(nn.Module):
    """Standard sinusoidal timestep embedding from original DDPM"""
    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t):
        # t shape: [Batch]
        half_dim = self.mlp[0].in_features // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return self.mlp(emb)

class ConditionalDiT(nn.Module):
    def __init__(self, latent_dim=256, context_dim=1280, num_blocks=6):
        super().__init__()
        self.time_embedder = TimestepEmbedder(latent_dim)
        self.affinity_embedder = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
        # Stack of DiT Blocks
        self.blocks = nn.ModuleList([
            ConditionalDiTBlock(hidden_dim=latent_dim, cond_dim=latent_dim, context_dim=context_dim)
            for _ in range(num_blocks)
        ])
        
        # Final projection to output noise
        self.final_norm = nn.LayerNorm(latent_dim)
        self.final_proj = nn.Linear(latent_dim, latent_dim)

    def forward(self, x, t, context, affinity):
        # x: [B, 256], t: [B], context: [B, 1280], affinity: [B, 1]
        
        t_emb = self.time_embedder(t)
        aff_emb = self.affinity_embedder(affinity)
        c_global = t_emb + aff_emb  # [B, 256]
        
        x = x.unsqueeze(1)          # [B, 1, 256]
        context = context.unsqueeze(1) # [B, 1, 1280]
        
        for block in self.blocks:
            x = block(x, c_global, context)
            
        x = x.squeeze(1) # Back to [B, 256]
        x = self.final_norm(x)
        return self.final_proj(x)