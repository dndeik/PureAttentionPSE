import math

import torch
from torch import nn as nn
from torch.nn import functional as F


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = (torch.rand(shape, device=x.device, dtype=x.dtype) < keep_prob).float()
        return x / keep_prob * random_tensor


class GQASelfAttentionRelPos(nn.Module):
    def __init__(self, embed_dim, num_heads, num_groups, max_position=128, bias=False, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        assert embed_dim % num_groups == 0
        assert num_heads % num_groups == 0

        self.embed_dim = embed_dim
        self.num_query_heads = num_heads
        self.num_kv_heads = num_groups
        self.head_dim = embed_dim // num_heads
        self.kv_repeat_factor = num_heads // num_groups
        self.max_position = max_position

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, self.head_dim * num_groups, bias=bias)
        self.v_proj = nn.Linear(embed_dim, self.head_dim * num_groups, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Rel Pos [-max_position+1, +max_position-1]
        self.rel_pos_emb = nn.Embedding(2 * max_position - 1, self.head_dim)

        self.pos_idx_cache = None

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None):
        """
        x: [B, T, D]
        attn_mask: [B * num_heads, T, T] (опционально)
        """
        bsz, tgt_len, _ = q.size()

        q = self.q_proj(q.transpose(0, 1)).view(tgt_len, bsz * self.num_query_heads, self.head_dim).transpose(0, 1)
        k_t = self.k_proj(k.transpose(0, 1)).view(tgt_len, bsz * self.num_kv_heads, self.head_dim).permute(1, 2, 0)
        v = self.v_proj(v.transpose(0, 1)).view(tgt_len, bsz * self.num_kv_heads, self.head_dim).transpose(0, 1)

        k_t = k_t.repeat_interleave(self.kv_repeat_factor, dim=0)
        v = v.repeat_interleave(self.kv_repeat_factor, dim=0)

        q_scaled = q * math.sqrt(1.0 / self.head_dim)

        attn_output_weights = torch.bmm(q_scaled, k_t)

        # Add rel pos
        rel_pos_bias = self.compute_rel_pos_bias(tgt_len, q.device)  # [T, T, head_dim]
        rel_bias_term = torch.einsum("bhd,thd->bht", q_scaled, rel_pos_bias)  # [B*H, T, T]
        attn_output_weights = attn_output_weights + rel_bias_term

        if attn_mask is not None:
            attn_output_weights = attn_output_weights.masked_fill(attn_mask, float("-inf"))

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = self.dropout(attn_output_weights)

        attn_output = torch.bmm(attn_output_weights, v)

        attn_output = (
            attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, self.embed_dim)
        )
        attn_output = self.out_proj(attn_output)
        attn_output = attn_output.view(tgt_len, bsz, self.embed_dim).transpose(0, 1)
        return attn_output

    def compute_rel_pos_bias(self, seq_len, device):
        """Создаёт матрицу релятивных позиций и возвращает эмбеддинги [T, T, head_dim]."""
        # Диапазон индексов [-T+1, T-1], сдвигаем чтобы не было отрицательных индексов
        if self.pos_idx_cache is None or self.pos_idx_cache.shape[1] != seq_len:
            range_vec = torch.arange(seq_len, device=device)
            distance_mat = range_vec[None, :] - range_vec[:, None]
            distance_mat_clamped = torch.clamp(distance_mat, -self.max_position + 1, self.max_position - 1)
            distance_mat_clamped = distance_mat_clamped + self.max_position - 1
            # [T, T, D_head]
            self.pos_idx_cache = distance_mat_clamped

        return self.rel_pos_emb(self.pos_idx_cache)


class GQASelfAttention2DRelPos(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        num_groups,
        max_time,
        max_freq,
        bias=False,
        dropout=0.0,
    ):
        super().__init__()

        assert embed_dim % num_heads == 0
        assert embed_dim % num_groups == 0
        assert num_heads % num_groups == 0

        self.embed_dim = embed_dim
        self.num_query_heads = num_heads
        self.num_kv_heads = num_groups
        self.head_dim = embed_dim // num_heads
        self.kv_repeat_factor = num_heads // num_groups

        self.max_time = max_time
        self.max_freq = max_freq

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, self.head_dim * num_groups, bias=bias)
        self.v_proj = nn.Linear(embed_dim, self.head_dim * num_groups, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # 2D relative bias
        self.rel_pos_time = nn.Embedding(2 * max_time - 1, self.head_dim)
        self.rel_pos_freq = nn.Embedding(2 * max_freq - 1, self.head_dim)

        # кэш
        self.rel_t_index = None
        self.rel_f_index = None

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None):
        """
        x: [B, T, D]
        attn_mask: [B * num_heads, T, T] (опционально)
        """
        bsz, tgt_len, _ = q.size()

        q = self.q_proj(q.transpose(0, 1)).view(tgt_len, bsz * self.num_query_heads, self.head_dim).transpose(0, 1)
        k_t = self.k_proj(k.transpose(0, 1)).view(tgt_len, bsz * self.num_kv_heads, self.head_dim).permute(1, 2, 0)
        v = self.v_proj(v.transpose(0, 1)).view(tgt_len, bsz * self.num_kv_heads, self.head_dim).transpose(0, 1)

        k_t = k_t.repeat_interleave(self.kv_repeat_factor, dim=0)
        v = v.repeat_interleave(self.kv_repeat_factor, dim=0)

        q_scaled = q * math.sqrt(1.0 / self.head_dim)

        attn_output_weights = torch.bmm(q_scaled, k_t)

        # Add rel pos
        rel_bias_term = self.compute_2d_relpos_bias(q_scaled, q.shape[1], k_t.shape[-1], self.max_freq)  # [B*H, T, T]
        attn_output_weights = attn_output_weights + rel_bias_term

        if attn_mask is not None:
            attn_output_weights = attn_output_weights.masked_fill(attn_mask, float("-inf"))

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = self.dropout(attn_output_weights)

        attn_output = torch.bmm(attn_output_weights, v)

        attn_output = (
            attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, self.embed_dim)
        )
        attn_output = self.out_proj(attn_output)
        attn_output = attn_output.view(tgt_len, bsz, self.embed_dim).transpose(0, 1)
        return attn_output

    def compute_2d_relpos_bias(self, q, q_len, k_len, freq_size):
        """
        q : [B*H, Nq, head_dim]

        возвращает:
        bias [B*H, Nq, Nk]
        """

        device = q.device

        rel_t, rel_f = self.build_relative_index(q_len, k_len, freq_size, device)  # [Nq, Nk]

        rel_emb_t = self.rel_pos_time(rel_t)  # [Nq, Nk, D]
        rel_emb_f = self.rel_pos_freq(rel_f)  # [Nq, Nk, D]

        rel_emb = rel_emb_t + rel_emb_f  # [Nq, Nk, D]

        bias = torch.einsum("bid,ijd->bij", q, rel_emb)

        return bias

    def build_relative_index(self, q_len, k_len, freq_size, device):
        """
        q_time — количество временных шагов у query
        k_time — количество временных шагов у key
        freq_size — число freq патчей

        Возвращает:
        rel_t, rel_f  [Nq, Nk]
        """

        q_time = q_len // self.max_freq
        k_time = k_len // self.max_freq

        N_q = q_time * freq_size
        N_k = k_time * freq_size

        if self.rel_t_index is None or self.rel_t_index.shape != (N_q, N_k):
            q_coords = torch.stack(
                torch.meshgrid(
                    torch.arange(q_time, device=device),
                    torch.arange(freq_size, device=device),
                    indexing="ij",
                )
            ).flatten(1)  # [2, Nq]

            k_coords = torch.stack(
                torch.meshgrid(
                    torch.arange(k_time, device=device),
                    torch.arange(freq_size, device=device),
                    indexing="ij",
                )
            ).flatten(1)  # [2, Nk]

            relative = q_coords[:, :, None] - k_coords[:, None, :]  # [2, Nq, Nk]

            delta_t = relative[0]
            delta_f = relative[1]

            delta_t = torch.clamp(delta_t, -self.max_time + 1, self.max_time - 1)
            delta_f = torch.clamp(delta_f, -self.max_freq + 1, self.max_freq - 1)

            self.rel_t_index = delta_t + self.max_time - 1
            self.rel_f_index = delta_f + self.max_freq - 1

        return self.rel_t_index, self.rel_f_index


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, factor=4, bias=True) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, 2 * factor * d_model, bias=bias)
        self.linear_2 = nn.Linear(factor * d_model, d_model, bias=bias)
        self.silu = nn.SiLU()

    def forward(self, x):
        x = self.linear_1(x)
        x1, x2 = torch.chunk(x, 2, -1)
        hidden = self.silu(x1) * x2
        return self.linear_2(hidden)
