import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class FeedForward(nn.Module):
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


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class SP_MSA(nn.Module):
    def __init__(self,
                 dim,
                 hdim,
                 heads=1,
                 qkv_bias=False,
                 qk_scale=None,
                 dropout=0.,
                 edge_attr=None,
                 ):
        super().__init__()

        assert dim % heads == 0, f"dim {dim} should be divided by num_heads {heads}."

        self.dima = 'dim'
        self.dim = dim
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.gat = GATConv(
            in_channels=dim,
            out_channels=dim,
            heads=4,
            concat=False,
            dropout=dropout,
            edge_dim=edge_attr if edge_attr is not None else None
        )

        # QKV 线性层
        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_linear = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        self.NL = PreNorm(dim, FeedForward(dim, hdim, dropout=dropout))

    def SpationAttention(self, x, edge_index, edge_attr):
        BT, N, C = x.shape
        device = x.device

        # Step 3: 使用 GAT 生成节点嵌入
        x_flat = x.reshape(-1, C)  # [B*T*N, C]
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device) if edge_attr is not None else None

        # 扩展 edge_index 和 edge_attr 以匹配批量
        edge_index_batch = torch.cat([
            edge_index + i * N for i in range(BT)
        ], dim=1)

        # GAT 前向传播
        gat_out = self.gat(x_flat, edge_index_batch)  #, edge_attr_batch [B*T*N, gat_hidden_dim]

        # QKV 生成
        q = self.q_linear(x).reshape(BT, N, self.num_heads, C // self.num_heads).permute(0, 2, 1,
                                                                                         3)  # [B*T, heads, N, C//heads]
        kv = self.kv_linear(gat_out).reshape(BT, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # [B*T, heads, N, C//heads]

        # Attention 权重
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B*T, heads, N, N]

        # 这里我们简化为无偏 attention
        attn = attn.softmax(dim=-1)

        # 注意力加权
        out = (attn @ v).transpose(1, 2).reshape(BT, N, C)
        out = self.proj(out)
        # out = self.proj_drop(out)
        return out

    def forward(self, x, edge_index, edge_attr):
        B, C, N, T = x.shape

        x = x.permute(0, 3, 2, 1).reshape(B * T, N, C)  # [B*T, N, C]

        sp_x = self.SpationAttention(x, edge_index, edge_attr)  # [B*T, 184, 32]
        x = sp_x + x  # [B*T, 184, 32]
        x = self.NL(x) + x  # [B*T, 184, 32]
        # 恢复原始形状 [B, C, N, T]
        x = x.view(B, T, N, C).permute(0, 3, 2, 1)  # [B, C, N, T]
        return x
