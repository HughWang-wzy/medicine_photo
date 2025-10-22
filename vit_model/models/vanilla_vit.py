import torch
from torch import nn

class PatchEmbedding(nn.Module):
    """将图像分割成块并进行线性嵌入"""
    def __init__(self, image_size, patch_size, in_channels, dim):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2
        self.patch_embedding = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

    def forward(self, x):
        B = x.shape[0]
        # 卷积 -> 展平 -> 调整维度
        x = self.patch_embedding(x).flatten(2).transpose(1, 2)
        # 添加CLS Token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # 添加位置编码
        x += self.pos_embedding
        return x

class Attention(nn.Module):
    """多头自注意力机制"""
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(t.shape[0], self.heads, -1, t.shape[-1] // self.heads).transpose(1, 2), qkv)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(x.shape[0], -1, self.heads * (x.shape[-1] // self.heads))
        return self.to_out(out)

class FeedForward(nn.Module):
    """MLP模块"""
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    """Transformer Encoder Block"""
    def __init__(self, dim, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x):
        x = self.attn(self.norm1(x)) + x
        x = self.ff(self.norm2(x)) + x
        return x

class ViT(nn.Module):
    """Vision Transformer"""
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, in_channels=3, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, dim)
        self.dropout = nn.Dropout(emb_dropout)
        
        self.transformer = nn.Sequential(*[
            TransformerBlock(dim, heads, mlp_dim, dropout) for _ in range(depth)
        ])
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.patch_embedding(img)
        x = self.dropout(x)
        x = self.transformer(x)
        
        # 只取CLS Token的输出用于分类
        cls_output = x[:, 0]
        return self.mlp_head(cls_output)
