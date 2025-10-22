import torch
from torch import nn
import torch.nn.functional as F
from .vanilla_vit import TransformerBlock, PatchEmbedding

# --- 方法一：DynamicViT ---
class TokenPruning(nn.Module):
    """一个简单的决策模块，用于Token剪枝"""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, x):
        # 输入x的形状: [B, N, D], N是token数量
        # 输出是每个token的重要性分数
        return self.net(x)

class DynamicViT(nn.Module):
    """
    实现了动态Token剪枝的ViT。
    """
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, 
                 pruning_loc, token_ratio, in_channels=3, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, dim)
        self.dropout = nn.Dropout(emb_dropout)
        
        self.depth = depth
        self.pruning_loc = pruning_loc
        self.token_ratio = token_ratio
        
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_dim, dropout) for _ in range(depth)
        ])
        
        self.pruners = nn.ModuleList([
            TokenPruning(dim) for _ in self.pruning_loc
        ])
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.patch_embedding(img)
        x = self.dropout(x)
        
        pruning_idx = 0
        for i, block in enumerate(self.blocks):
            x = block(x)
            
            # 在指定位置进行剪枝
            if (i + 1) in self.pruning_loc:
                pruner = self.pruners[pruning_idx]
                pruning_idx += 1
                
                cls_token, tokens = x[:, :1], x[:, 1:]
                B, N, D = tokens.shape
                
                # 计算重要性分数并决定保留哪些token
                scores = pruner(tokens).squeeze(-1) # [B, N-1]
                num_keep = int(N * self.token_ratio)
                
                # 每个样本独立排序和选择
                indices_to_keep = torch.topk(scores, k=num_keep, dim=1).indices # [B, num_keep]
                
                # 使用gather选择token
                # 为indices增加一个维度以匹配tokens的维度
                indices_to_keep = indices_to_keep.unsqueeze(-1).expand(-1, -1, D)
                tokens = torch.gather(tokens, 1, indices_to_keep)
                
                # 重新组合CLS token和被保留的图像token
                x = torch.cat([cls_token, tokens], dim=1)

        cls_output = x[:, 0]
        return self.mlp_head(cls_output)


# --- 方法二：HierarchicalViT ---
class TokenMerging(nn.Module):
    """
    Token融合模块，用于降低序列长度，增加维度。
    将 2x2 的token融合成一个。
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(4 * in_dim)
        self.reduction = nn.Linear(4 * in_dim, out_dim, bias=False)

    def forward(self, x):
        B, L, C = x.shape
        H = W = int(L**0.5)
        
        x = x.view(B, H, W, C)
        
        # 将 2x2 的patch展平
        p1 = x[:, 0::2, 0::2, :] # [B, H/2, W/2, C]
        p2 = x[:, 1::2, 0::2, :]
        p3 = x[:, 0::2, 1::2, :]
        p4 = x[:, 1::2, 1::2, :]
        
        x = torch.cat([p1, p2, p3, p4], dim=-1) # [B, H/2, W/2, 4*C]
        x = x.view(B, -1, 4 * C) # [B, L/4, 4*C]
        
        x = self.norm(x)
        x = self.reduction(x)
        
        return x

class HierarchicalViT(nn.Module):
    """
    实现了分层结构和Token融合的ViT。
    """
    def __init__(self, image_size, patch_size, num_classes, dims, depths, heads, in_channels=3):
        super().__init__()
        
        # 初始的Patch嵌入
        initial_dim = dims[0]
        self.patch_embed = nn.Conv2d(in_channels, initial_dim, kernel_size=patch_size, stride=patch_size)
        
        num_patches = (image_size // patch_size)**2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, initial_dim))
        
        self.stages = nn.ModuleList()
        num_stages = len(dims)

        for i in range(num_stages):
            # 当前阶段的Transformer Blocks
            stage_blocks = nn.Sequential(*[
                TransformerBlock(dims[i], heads[i], dims[i] * 4) for _ in range(depths[i])
            ])
            self.stages.append(stage_blocks)
            
            # 如果不是最后一个阶段，添加Token融合层
            if i < num_stages - 1:
                merger = TokenMerging(dims[i], dims[i+1])
                self.stages.append(merger)

        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, img):
        x = self.patch_embed(img).flatten(2).transpose(1, 2)
        x += self.pos_embed
        
        for stage in self.stages:
            x = stage(x)
            
        # 使用全局平均池化代替CLS Token
        x = self.norm(x)
        x = x.mean(dim=1)
        
        return self.head(x)
