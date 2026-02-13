import torch 
import torch.nn as nn

#define embedding layer
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_channels=1, embed_dim=128):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size //patch_size) ** 2

    def forward(self, x):
        x = self.proj(x) #(B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2) #(B, N, D)
        return x

#single transformer encoder
class CustomEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.attn_weights = None  # store attention

    def forward(self, x):
        attn_out, attn_weights = self.attn(x, x, x, need_weights=True)
        self.attn_weights = attn_weights  # (B, heads, N, N)

        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        mlp_out = self.mlp(x)
        x = x + self.dropout(mlp_out)
        x = self.norm2(x)

        return x
    
#define vision transformer model
class ViT(nn.Module):
    def __init__(
            self,
            img_size=64,
            patch_size=8,
            in_channels=1,
            num_classes=4,
            embed_dim=128,
            depth=6,
            num_heads=8,
            mlp_dim=256,
            dropout=0.1
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter( torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim))
        
        self.encoder_layers = nn.ModuleList([
            CustomEncoderLayer(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) #define final classifier layer

    def forward(self, x):
        B = x.size(0)

        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1) #(B, 1, D)

        x = torch.cat((cls_tokens, x), dim=1) #(B, 1+N, D)
        x = x + self.pos_embed

        attn_weights_all = [] #save all the weights

        for layer in self.encoder_layers:
            x = layer(x)
            attn_weights_all.append(layer.attn_weights)

        x = self.norm(x)
        cls_out = x[:, 0]
        logits = self.head(cls_out)

        return logits, attn_weights_all
