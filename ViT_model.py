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
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True
        ) #define single transformer encoder layer

        self.encoder = nn.TransformerEncoder(encoder_layer, depth) #6 layers of transformer layers
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) #define final classifier layer

    def forward(self, x):
        B = x.size(0)

        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1) #(B, 1, D)

        x = torch.cat((cls_tokens, x), dim=1) #(B, 1+N, D)
        x = x + self.pos_embed

        x = self.encoder(x)
        x = self.norm(x)

        cls_out = x[:, 0]
        return self.head(cls_out)