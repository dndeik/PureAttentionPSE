import torch
import torch.nn as nn
from einops import rearrange

from model.full_model.transformer_modules import DropPath, SwiGLUFFN
from model.stream_model.stream_transformer_modules import StreamGQASelfAttention2DRelPos
from model.CONSTANTS import GLOBAL_EPS

from model.utils import count_parameters


class StreamTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, chunk_size, left_context, max_time, max_freq, dropout=0., drop_path=0.):
        super().__init__()
        self.chunk_size = chunk_size
        self.left_context = left_context
        self.pre_norm = nn.LayerNorm(d_model, eps=GLOBAL_EPS)

        self.attn = StreamGQASelfAttention2DRelPos(d_model,
                                                   num_heads=num_heads,
                                                   num_groups=num_heads // 2,
                                                   max_freq=max_freq,
                                                   max_time=max_time,
                                                   dropout=dropout)

        self.attn_res_coef = nn.Parameter(torch.ones(1))

        self.inter_norm = nn.LayerNorm(d_model, eps=GLOBAL_EPS)
        self.ffn = SwiGLUFFN(d_model)
        self.ffn_res_coef = nn.Parameter(torch.ones(1))

        self.drop_path = DropPath(drop_path)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_cache_k, attn_cache_v):
        residual = x
        x = self.pre_norm(x)

        x, attn_cache_k, attn_cache_v = self.attn(x, x, x, attn_cache_k, attn_cache_v)
        x = residual + self.drop_path(self.attn_res_coef *
                                      self.dropout(x))

        residual = x
        x = self.inter_norm(x)
        x = residual + self.drop_path(self.ffn_res_coef * self.dropout(self.ffn(x)))

        return x, attn_cache_k, attn_cache_v


class StreamTimeAttn(nn.Module):
    def __init__(self,
                 attn_dim,
                 input_channels,
                 output_channels,
                 time_chunk_size,
                 time_patch_size,
                 freq_size,
                 freq_patch_size,
                 context_chunk_number=3,
                 num_heads=4,
                 layer_num=1,
                 dropout=0.1,
                 drop_path=0.1):
        super().__init__()
        self.freq_patch_size = freq_patch_size
        self.time_patch_size = time_patch_size

        self.max_freq = freq_size // self.freq_patch_size
        time_in_chunk = time_chunk_size // self.time_patch_size
        self.token_in_chunk = self.max_freq * time_in_chunk
        self.max_time = time_in_chunk * (context_chunk_number + 1)
        self.left_context = int(self.token_in_chunk * context_chunk_number)
        self.in_linear_dim = input_channels * self.freq_patch_size * self.time_patch_size
        self.out_linear_dim = output_channels * self.freq_patch_size * self.time_patch_size

        self.in_proj = nn.Linear(self.in_linear_dim, attn_dim)

        self.blocks = nn.ModuleList(
            [StreamTransformerBlock(attn_dim,
                                    num_heads=num_heads,
                                    chunk_size=self.token_in_chunk,
                                    left_context=self.left_context,
                                    max_time=self.max_time,
                                    max_freq=self.max_freq,
                                    dropout=dropout,
                                    drop_path=drop_path) for _ in
             range(layer_num)])

        self.out_proj = nn.Linear(attn_dim, self.out_linear_dim)

    def forward(self, x, attn_cache_k, attn_cache_v):
        # [B, C, T, F]
        B, C, T, F = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().reshape(B, -1, self.time_patch_size, F,
                                                       C)  # [B, T//time_patch, time_patch, F, C]
        x = rearrange(x, "b time p1 (f p2) c -> b time f (p1 p2 c)", p1=self.time_patch_size, p2=self.freq_patch_size)
        token_in_time_step = x.shape[-2]
        x = x.reshape(B, -1, x.shape[-1])
        x = self.in_proj(x)

        for idx, block in enumerate(self.blocks):
            x, attn_cache_k[idx], attn_cache_v[idx] = block(x, attn_cache_k[idx], attn_cache_v[idx])

        x = self.out_proj(x)
        x = x.reshape(B, -1, token_in_time_step, self.out_linear_dim)
        x = rearrange(x, " b time f (p1 p2 c) -> b time p1 (f p2) c", p1=self.time_patch_size,
                      p2=self.freq_patch_size)
        x = x.reshape(B, T, F, -1).permute(0, 3, 1, 2)

        return x, attn_cache_k, attn_cache_v


if __name__ == "__main__":
    inter_d_model = 256
    num_heads = 8
    conv_channels = 96
    freq_size = 161
    freq_patch_size = 23
    time_chunk_size = 8
    time_patch_size = 2
    layer_num = 1
    left_context_chunk_num = 2
    model = StreamTimeAttn(inter_d_model,
                           input_channels=conv_channels,
                           output_channels=conv_channels,
                           freq_size=freq_size,
                           freq_patch_size=freq_patch_size,
                           time_chunk_size=time_chunk_size,
                           time_patch_size=time_patch_size,
                           layer_num=layer_num,
                           context_chunk_number=left_context_chunk_num).eval()
    count_parameters(model)

    x = torch.randn(1, conv_channels, 400, freq_size)

    head_dim = inter_d_model // num_heads
    left_context = (freq_size // freq_patch_size) * (time_chunk_size // time_patch_size) * left_context_chunk_num
    half_num_heads = num_heads // 2
    double_head_dim = head_dim * 2
    enc_attn_cache_k = torch.zeros(layer_num, half_num_heads, double_head_dim, left_context)
    enc_attn_cache_v = torch.zeros(layer_num, half_num_heads, left_context, double_head_dim)

    print("INPUT SHAPE: ", x.shape)
    outputs = []
    for i in range(0, x.shape[-2], time_chunk_size):
        cur_chunk = x[..., i: i+time_chunk_size, :]
        out, enc_attn_cache_k, enc_attn_cache_v = model(cur_chunk, enc_attn_cache_k, enc_attn_cache_v)
        outputs.append(out)

    full_out = torch.cat(outputs, dim=2)
    print("OUTPUT SHAPE: ", full_out.shape)
