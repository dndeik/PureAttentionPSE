import torch
import torch.nn as nn
from einops import rearrange

from model.full_model.transformer_modules import DropPath, GQASelfAttention2DRelPos, SwiGLUFFN
from model.CONSTANTS import GLOBAL_EPS


class TransformerBlock(nn.Module):
    def __init__(self, d_model, time_size, freq_size, num_heads, dropout=0., drop_path=0.):
        super().__init__()
        self.pre_norm = nn.LayerNorm(d_model, eps=GLOBAL_EPS)
        self.attn = GQASelfAttention2DRelPos(d_model,
                                             num_heads=num_heads,
                                             num_groups=num_heads // 2,
                                             max_time=time_size,
                                             max_freq=freq_size,
                                             dropout=dropout)
        self.attn_res_coef = nn.Parameter(torch.ones(1))

        self.inter_norm = nn.LayerNorm(d_model, eps=GLOBAL_EPS)
        self.ffn = SwiGLUFFN(d_model)
        self.ffn_res_coef = nn.Parameter(torch.ones(1))

        self.drop_path = DropPath(drop_path)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        residual = x
        x = self.pre_norm(x)
        x = residual + self.drop_path(self.attn_res_coef * self.dropout(self.attn(x, x, x, attn_mask=attn_mask)))

        residual = x
        x = self.inter_norm(x)
        x = residual + self.drop_path(self.ffn_res_coef * self.dropout(self.ffn(x)))

        return x


class IntraChunkAttn(nn.Module):
    def __init__(self,
                 attn_dim,
                 input_channels,
                 time_chunk_size,
                 freq_dim,
                 freq_patch_size,
                 num_heads=4,
                 layer_num=1,
                 remain_dim=True,
                 dropout=0.1,
                 drop_path=0.1):

        super().__init__()
        self.remain_dim = remain_dim
        self.time_chunk_size = time_chunk_size
        self.freq_patch_size = freq_patch_size
        self.num_heads = num_heads

        self.in_proj = nn.Linear(input_channels * self.freq_patch_size, attn_dim, bias=False)

        self.blocks = nn.Sequential(*[TransformerBlock(attn_dim,
                                                       time_size=self.time_chunk_size,
                                                       freq_size=freq_dim//freq_patch_size,
                                                       num_heads=self.num_heads,
                                                       dropout=dropout,
                                                       drop_path=drop_path) for _ in range(layer_num)])

        if self.remain_dim:
            self.out_proj = nn.Linear(attn_dim, input_channels * self.freq_patch_size)

        self.attn_drop_p = drop_path

    def _random_attention_mask(self, B: int, L: int, device=None):
        """
        Возвращает булеву маску (L, L),
        где True = замаскировано.

        - В eval режиме возвращает None.
        - Диагональ всегда False.
        """

        if not self.training or self.attn_drop_p == 0.0:
            return None

        p = self.attn_drop_p

        if not 0.0 <= p < 1.0:
            raise ValueError("attn_drop_p must be in [0, 1).")

        # Семплируем случайную маску
        mask = torch.rand(B, L, L, device=device) < p  # True = mask

        # Диагональ всегда разрешена
        diag_idx = torch.arange(L, device=device)
        mask[:, diag_idx, diag_idx] = False

        mask = torch.repeat_interleave(mask, self.num_heads, dim=0)

        return mask

    def forward(self, x):  # [B, C, T, F]
        B, C, T, F = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().reshape(B, -1, self.time_chunk_size, F, C)  # [B, T//chunk_size, chunk_size, F, C]
        x = x.reshape(-1, self.time_chunk_size, F, C) # [B', T', F, C]
        x = rearrange(x, "b t (f p) c -> b (t f) (p c)", p=self.freq_patch_size)
        x = self.in_proj(x)

        random_mask = self._random_attention_mask(x.shape[0], x.shape[1], x.device)
        for block in self.blocks:
            x = block(x, random_mask)

        if self.remain_dim:
            x = self.out_proj(x)
            x = rearrange(x, "b (t f) (p c) -> b t (f p) c", t=self.time_chunk_size, p=self.freq_patch_size)
            x = x.reshape(B, T, F, C).permute(0, 3, 1, 2)

        return x
