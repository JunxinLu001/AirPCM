import torch
import torch.nn as nn
from einops import rearrange
from models.SMSA import SP_MSA
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class LearnablePatchPositionEmbedding(nn.Module):
    def __init__(self, num_patches, d_model):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model))

    def forward(self, x):
        return x + self.pos_embed[:, :x.size(1), :]


class FineGrainedCausalAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q_emb, kv_emb, attn_mask=None):
        # q_emb: [B*N, Qlen, d_model]
        # kv_emb: [B*N, L, V, d_model]
        Bn, Qlen, _ = q_emb.shape
        L, V = kv_emb.shape[1], kv_emb.shape[2]

        K = self.wk(kv_emb).view(Bn, L * V, self.n_heads, self.head_dim).transpose(1, 2)  # [Bn, h, L*V, Hd]
        Vv = self.wv(kv_emb).view(Bn, L * V, self.n_heads, self.head_dim).transpose(1, 2)
        Q = self.wq(q_emb).view(Bn, Qlen, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [Bn, h, Qlen, L*V]
        if attn_mask is not None:
            # attn_mask: [Bn, 1, Qlen, L, V] or broadcastable -> flatten to [Bn,1,Qlen,L*V]
            mask = attn_mask.reshape(1, 1, Qlen, L * V)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, Vv)  # [Bn, h, Qlen, Hd]
        out = out.transpose(1, 2).reshape(Bn, Qlen, self.d_model)
        return self.out(out), attn.view(Bn, self.n_heads, Qlen, L, V)


class SpatioTemporalBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.temporal = TransformerEncoderLayer(d_model, nhead, d_model * 2, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.temporal(x)
        return self.norm(x)


class VariableAdapter(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.down = nn.Linear(d_model, d_model // 2)
        self.act = nn.ReLU()
        self.up = nn.Linear(d_model // 2, d_model)

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.pred_len = configs.pred_len  # 预测长度
        self.seq_len = configs.seq_len  # 输入时序长度
        self.patch_size = configs.patch_size  # path大小 4
        self.stride = configs.stride
        self.d_model = configs.d_model
        self.nv = configs.number_variable
        self.hidden_channels = configs.hidden_channels
        self.tv = configs.pred_variable
        self.sn = configs.station_num
        self.bt = configs.batch_size
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1
        self.patch_num += 1
        self.device = configs.device

        # 历史气象时延窗口长度 L
        self.lag_window = configs.lag_window  # 例如 8 或 12

        self.conv1 = nn.Conv2d(self.nv, self.hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.hidden_channels, self.nv, kernel_size=3, padding=1)
        self.SMSA_blocks = nn.Sequential(*[
            SP_MSA(self.hidden_channels, self.hidden_channels * 2, heads=configs.head,
                   dropout=configs.dropout)  # , edge_attr=2
            for _ in range(2)
        ])

        self.padding = nn.ReplicationPad1d((0, self.stride))
        self.enc_embedding = nn.Linear(self.patch_size, self.hidden_channels)
        self.pos_encoder = LearnablePatchPositionEmbedding(self.patch_num, self.hidden_channels)
        # 一次性映射所有气象通道到多头 embedding
        self.meteo_linear = nn.Linear(self.nv - self.tv, (self.nv - self.tv) * self.hidden_channels)

        self.fine_attn = FineGrainedCausalAttention(self.hidden_channels, n_heads=4)

        self.fusion = nn.Linear(2 * self.hidden_channels, self.d_model)

        self.temporal_blocks = nn.Sequential(*[
            SpatioTemporalBlock(self.d_model, nhead=8, dropout=0.1) for _ in range(3)
        ])

        self.adapters = nn.ModuleList([VariableAdapter(self.d_model) for _ in range(self.tv)])
        self.out_layers = nn.ModuleList([
            nn.Linear(self.d_model * self.patch_num, self.pred_len) for _ in range(self.tv)
        ])

    def get_patch(self, x):
        if torch.isnan(x).any():
            print("NaN detected in input before unfold!")
        x = self.padding(x)  #  [1,16,360]
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, 'b m n p -> (b m) n p')
        return x

    def forward(self, inputs, edge_index, edge_attr):
        B, T, N, D = inputs.shape  # [B, T, N, D]
        x = inputs.permute(0, 3, 2, 1)  # [B, T, N, D] -> [B, D, N, T]
        # 进行卷积操作
        x = self.conv1(x)  # [B, D, N, T] -> [B, H, N, T]
        for blk in self.SMSA_blocks:
            x = blk(x, edge_index, edge_attr)  # [B, H, N, T]
        x_sp = self.conv2(x)  # [B, D, N, T]

        # 拆分出空气与气象
        air, meteo = x_sp[:, :self.tv], x_sp[:, self.tv:]

        all_out, all_attn = [], []

        for idx in range(self.tv):
            var_i = air[:, idx].squeeze(1)  # [B, N, T]
            patch = self.get_patch(var_i)  # [B*N, patch_num, patch_size]
            q = self.enc_embedding(patch)  # [B*N, num_patch, H]
            q = self.pos_encoder(q)  # [B*N, num_patch, H]

            # 按当前时刻向前取 lag_window
            met = meteo[:, :, :, -self.lag_window:]  # [B, V, N, L]
            met = met.permute(0, 2, 3, 1).reshape(B * N, self.lag_window, self.nv - self.tv)  # [B*N, L, V]

            # 一次性映射所有气象变量
            flat = self.meteo_linear(met)  # [B*N, L, V*H]
            kv = flat.view(B * N, self.lag_window, self.nv - self.tv, self.hidden_channels)

            mask = torch.tril(torch.ones(self.patch_num, self.lag_window, self.nv - self.tv,
                                         device=met.device)).bool()  # [P, L, V]
            mask = mask.unsqueeze(0).unsqueeze(1)  # [1,1,P,L,V]

            tc_out, attn_w = self.fine_attn(q, kv, attn_mask=mask)

            # att_w 集成
            attn_w = torch.mean(attn_w, dim=1)
            attn_w = torch.max(attn_w, dim=2)[0]
            all_attn.append(attn_w.detach().cpu())

            fusion = self.fusion(torch.cat([q, tc_out], dim=-1))  # [BN, P, H]

            out_t = fusion.permute(1, 0, 2)
            for blk in self.temporal_blocks:
                out_t = blk(out_t)
            out_t = out_t.permute(1, 0, 2)

            adapted = self.adapters[idx](out_t)
            flat_o = adapted.reshape(B * N, -1)
            y = self.out_layers[idx](flat_o).view(B, N, -1).permute(0, 2, 1)
            all_out.append(y)

        out = torch.stack(all_out, dim=-1)
        all_attn = torch.stack(all_attn, dim=-1)
        all_attn = rearrange(all_attn, '(b n) p v mv-> b n p v mv', b=B, n=N)
        return out, all_attn
