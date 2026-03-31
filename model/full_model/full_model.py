import torch
import torch.nn as nn
import torch.nn.functional as F

from model.full_model.transformer_modules import GQASelfAttentionRelPos, DropPath
from model.full_model.time_attn import TimeAttn

from model.utils import count_parameters


class SeparableConv2d(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
    ):
        super().__init__()
        self.in_channels = in_channels,
        self.out_channels = out_channels,
        self.kernel_size = kernel_size,
        self.stride = stride,
        self.padding = padding,
        self.dilation = dilation,
        self.bias = bias,

        self.depthwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        self.pointwise_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias,
        )

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class CausalConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, k_size=3, conv_style="full"):
        super().__init__()

        if conv_style == "full":
            conv_type = nn.Conv2d
        elif conv_style == "dw":
            conv_type = SeparableConv2d
        else:
            NotImplementedError("Unknown conv_type")

        self.t_pad = k_size - 1
        self.f_pad = k_size // 2
        self.conv_padding = nn.ConstantPad2d((0, 0, self.t_pad, 0), 0)
        self.conv = conv_type(in_channel, out_channel, k_size, stride=(1, 1), padding=(0, self.f_pad))

    def forward(self, x):
        """[B, 2, T, F]"""
        x = self.conv_padding(x)
        out = self.conv(x)  # [B, inter_channel, T, F//2]
        return out


class SpeechDecoder(nn.Module):
    def __init__(self, inter_channel, k_size=3):
        super().__init__()
        self.t_pad = k_size - 1
        self.f_pad = k_size // 2
        self.conv_padding = nn.ConstantPad2d((0, 0, self.t_pad, 0), 0)

        # self.decoder_conv = nn.ConvTranspose2d(inter_channel, 2, k_size, stride=(1, 1), padding=(0, self.f_pad), bias=False)
        self.deconv = nn.Conv2d(inter_channel, 2, k_size, stride=(1, 1), padding=(0, self.f_pad), bias=False)

    def forward(self, x):
        x = self.conv_padding(x)
        # out = self.decoder_conv(x)[..., self.t_pad:-self.t_pad, :]  # [B, 2, T, F]
        out = self.deconv(x)  # [B, 2, T, F]
        return out


class CRM(nn.Module):
    """Faster Complex Ratio Mask"""

    def __init__(self):
        super().__init__()

    def forward(self, mask, spec):
        a = spec[:, 0]
        b = spec[:, 1]

        c = mask[:, 0]
        d = mask[:, 1]

        k1 = c * (a + b)
        k2 = a * (d - c)
        k3 = b * (c + d)

        s_real = k1 - k3
        s_imag = k1 + k2
        s = torch.stack([s_real, s_imag], dim=1)  # (B,2,T,F)
        return s


class FusionModule(nn.Module):
    def __init__(self, emb_dim, attn_dim, time_chunk_size, num_heads=4, dropout=0.1, drop_path=0.):
        super().__init__()
        self.linear = nn.Linear(emb_dim, attn_dim, bias=False)

        self.attn = GQASelfAttentionRelPos(attn_dim,
                                           num_heads=num_heads,
                                           num_groups=num_heads,
                                           dropout=dropout,
                                           )

        self.time_chunk_size = time_chunk_size

        self.fusion = nn.Conv2d(attn_dim * 2, attn_dim, kernel_size=1)
        self.mask = None

        self.drop_path = DropPath(drop_path)
        self.dropout = nn.Dropout(dropout)

    def get_mask(self, chunk_size, mask_length, left_context=0):
        if self.mask is None or mask_length != self.mask.size()[1]:
            mask = torch.zeros(mask_length, mask_length + left_context)
            chunk = torch.ones(chunk_size, chunk_size + left_context)
            for i in range(0, mask_length, chunk_size):
                mask[i:i + chunk_size, i:i + chunk_size + left_context] = chunk

            mask = mask[:, left_context:]

            self.mask = ~mask.bool()
        return self.mask

    def forward(self,
                esti: torch.Tensor,
                aux: torch.Tensor) -> torch.Tensor:  # [B, C, T, F]
        B, C, T, F = esti.shape
        esti_flatten = esti.permute(0, 3, 2, 1).reshape(B * F, T, C)  # [B*F, T, C]

        aux = self.linear(self.dropout(aux))
        aux = torch.repeat_interleave(aux, F, dim=0)  # [B*F, C]
        aux = aux.unsqueeze(1).repeat(1, T, 1)  # [B*F, T, C]

        mask = self.get_mask(self.time_chunk_size, esti_flatten.shape[1]).to(esti_flatten.device)
        aux_adapt = self.attn(aux, esti_flatten, esti_flatten, attn_mask=mask)
        aux = aux + self.drop_path(self.dropout(aux_adapt))  # [B*F, T, C]

        aux = aux.reshape(B, F, T, C).permute(0, 3, 2, 1)
        esti = self.fusion(torch.cat((esti, aux), dim=1))  # [B, C, T, F]
        return esti


class PureTransformerPSE(nn.Module):
    def __init__(self,
                 attn_dim=384,
                 embedding_dim=256,
                 fusion_dim=128,
                 enc_conv_channels=16,
                 dec_conv_channels=24,
                 time_chunk_size=8,
                 time_patch_size=2,
                 freq_dim=161,
                 freq_patch_size=23,
                 left_context_chunk_number=3,
                 enc_layer_num=8,
                 dec_layer_num=9,
                 num_heads=12,
                 dropout=0.15,
                 drop_path=0.1):
        super().__init__()

        #____________ENCODER_________________
        self.input_conv = CausalConv2d(2, enc_conv_channels)
        self.encoder = TimeAttn(attn_dim=attn_dim,
                                input_channels=enc_conv_channels,
                                output_channels=enc_conv_channels,
                                time_chunk_size=time_chunk_size,
                                time_patch_size=time_patch_size,
                                freq_size=freq_dim,
                                freq_patch_size=freq_patch_size,
                                context_chunk_number=left_context_chunk_number,
                                num_heads=num_heads,
                                layer_num=enc_layer_num,
                                dropout=dropout,
                                drop_path=drop_path)

        # ____________FUSION_________________
        self.up_conv = CausalConv2d(enc_conv_channels, fusion_dim)
        self.fusion_block = FusionModule(embedding_dim, fusion_dim, time_chunk_size, dropout=dropout,
                                         drop_path=drop_path)
        self.down_conv = CausalConv2d(fusion_dim, dec_conv_channels)

        # ____________DECODER_________________
        self.decoder = TimeAttn(attn_dim=attn_dim,
                                input_channels=dec_conv_channels,
                                output_channels=2,
                                time_chunk_size=time_chunk_size,
                                time_patch_size=time_patch_size,
                                freq_size=freq_dim,
                                freq_patch_size=freq_patch_size,
                                context_chunk_number=left_context_chunk_number,
                                num_heads=num_heads,
                                layer_num=dec_layer_num,
                                dropout=dropout,
                                drop_path=drop_path)
        self.crm = CRM()

    def forward(self, x, embedding):
        embedding = F.normalize(embedding, p=2, dim=-1)

        src_signal = x
        x = self.input_conv(x)
        x = self.encoder(x)
        x = self.up_conv(x)
        x = self.fusion_block(x, embedding)
        x = self.down_conv(x)
        x = self.decoder(x)
        x = self.crm(x, src_signal)

        return x


if __name__ == "__main__":
    conv_channels = 16
    freqs = 161
    chunk_size = 8
    emb_dim = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PureTransformerPSE(attn_dim=384, num_heads=6, embedding_dim=256, left_context_chunk_number=3, dropout=0.2,
                               drop_path=0.15)  # good start point
    model.to(device=device)
    count_parameters(model)

    bs = 2
    emb = torch.randn(bs, emb_dim, device=device)
    x = torch.randn(bs, 2, 200, freqs, device=device)
    print(x.shape)
    full_out = model(x, emb)
    print(full_out.shape)
