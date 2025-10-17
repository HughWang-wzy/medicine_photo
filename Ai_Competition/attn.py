from torch import nn
from einops import rearrange

class MultiheadSelfAttn(nn.Module):
    def __init__(self, dmodel, nhead, qk_norm, kv_bias, dropout):
        assert dmodel % nhead == 0
        # self.dhead = dmodel // nhead
        self.nhead = nhead
        self.scale = (dmodel // nhead) ** -0.5

        self.q_proj = nn.Linear(dmodel, dmodel, False)
        self.k_proj = nn.Linear(dmodel, dmodel, kv_bias)
        self.v_proj = nn.Linear(dmodel, dmodel, kv_bias)

        self.q_norm = nn.LayerNorm(dmodel, bias=False) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(dmodel, bias=False) if qk_norm else nn.Identity()

        self.end_proj = nn.Linear(dmodel, dmodel, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        res = x
        q = rearrange(self.q_norm(self.q_proj(x)), 'b l (n d) -> b n l d', n=self.nhead)
        k = rearrange(self.k_norm(self.k_proj(x)), 'b l (n d) -> b n d l', n=self.nhead)
        v = rearrange(self.v_proj(x), 'b l (n d) -> b n l d')
        
        x = self.dropout(((q @ k) * self.scale).softmax(dim=-2) @ v)
        x = rearrange(x, 'b n l d -> b l (n d)')
        x = self.end_proj(x)
        return x + res