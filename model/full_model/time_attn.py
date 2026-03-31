import torch
import torch.nn as nn
from einops import rearrange

from model.full_model.transformer_modules import DropPath, GQASelfAttention2DRelPos, SwiGLUFFN
from model.CONSTANTS import GLOBAL_EPS


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, chunk_size, left_context, max_time, max_freq, dropout=0., drop_path=0.):
        super().__init__()
        self.chunk_size = chunk_size
        self.left_context = left_context
        self.pre_norm = nn.LayerNorm(d_model, eps=GLOBAL_EPS)

        self.attn = GQASelfAttention2DRelPos(d_model,
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

    def forward(self, x, attn_mask=None):
        B, L, D = x.shape

        residual = x
        x = self.pre_norm(x)
        pad_tensor = torch.zeros((B, self.left_context, D), device=x.device, dtype=x.dtype)
        padded_x = torch.cat((pad_tensor, x), dim=1)
        x = residual + self.drop_path(self.attn_res_coef *
            self.dropout(self.attn(padded_x, padded_x, padded_x, attn_mask=attn_mask)[:, self.left_context:, :]))

        residual = x
        x = self.inter_norm(x)
        x = residual + self.drop_path(self.ffn_res_coef * self.dropout(self.ffn(x)))

        return x


class TimeAttn(nn.Module):
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
            [TransformerBlock(attn_dim,
                              num_heads=num_heads,
                              chunk_size=self.token_in_chunk,
                              left_context=self.left_context,
                              max_time=self.max_time,
                              max_freq=self.max_freq,
                              dropout=dropout,
                              drop_path=drop_path) for _ in
             range(layer_num)])

        self.out_proj = nn.Linear(attn_dim, self.out_linear_dim)

        self.mask = None

    def get_mask(self, chunk_size, mask_length, left_context=0, device=torch.device('cpu')):
        if self.mask is None or mask_length != self.mask.size()[1]:
            mask = torch.zeros(mask_length, mask_length + left_context, device=device)
            chunk = torch.ones(chunk_size, chunk_size + left_context, device=device)
            for i in range(0, mask_length, chunk_size):
                mask[i:i + chunk_size, i:i + chunk_size + left_context] = chunk

            mask = mask[:, left_context:]

            self.mask = ~mask.bool()
        return self.mask

    def forward(self, x):
        # [B, C, T, F]

        B, C, T, F = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().reshape(B, -1, self.time_patch_size, F, C)  # [B, T//time_patch, time_patch, F, C]
        x = rearrange(x, "b time p1 (f p2) c -> b time f (p1 p2 c)", p1=self.time_patch_size, p2=self.freq_patch_size)
        token_in_time_step = x.shape[-2]
        x = x.reshape(B, -1, x.shape[-1])
        x = self.in_proj(x)

        self.mask = self.get_mask(self.token_in_chunk, x.shape[1]+self.left_context, self.left_context, device=x.device)
        for block in self.blocks:
            x = block(x, self.mask)

        x = self.out_proj(x)
        x = x.reshape(B, -1, token_in_time_step, self.out_linear_dim)
        x = rearrange(x, " b time f (p1 p2 c) -> b time p1 (f p2) c", p1=self.time_patch_size,
                        p2=self.freq_patch_size)
        x = x.reshape(B, T, F, -1).permute(0, 3, 1, 2)

        return x


def count_parameters(model):
    parametrs_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {parametrs_num} ({parametrs_num / (10 ** 6):.1f}M)")


if __name__ == "__main__":
    chunk_size = 8
    conv_channels = 96
    inter_d_model = 256
    freqs = 161
    freq_patch_size = 23
    model = TimeAttn(inter_d_model, conv_channels, freq_size=freqs, freq_patch_size=freq_patch_size, time_chunk_size=8, time_patch_size=2,
                     layer_num=1).eval()
    count_parameters(model)

    x = torch.randn(1, conv_channels, 400, freqs)
    print(x.shape)
    full_out = model(x)
    print(full_out.shape)
