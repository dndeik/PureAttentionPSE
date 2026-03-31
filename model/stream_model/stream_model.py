import torch
import torch.nn as nn
import torch.nn.functional as F

from model.full_model.full_model import SeparableConv2d, FusionModule, CRM
from model.stream_model.stream_time_attn import StreamTimeAttn


class StreamCausalConv2d(nn.Module):
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
        self.conv = conv_type(in_channel, out_channel, k_size, stride=(1, 1), padding=(0, self.f_pad))

    def forward(self, x, conv_cache):
        x_with_cache = torch.cat([conv_cache, x], dim=2)
        conv_cache = x_with_cache[..., -self.t_pad:, :]
        x = self.conv(x_with_cache)
        return x, conv_cache


class StreamSpeechDecoder(nn.Module):
    def __init__(self, inter_channel, k_size=3):
        super().__init__()
        self.t_pad = k_size - 1
        self.f_pad = k_size // 2

        # self.decoder_conv = nn.ConvTranspose2d(inter_channel, 2, k_size, stride=(1, 1), padding=(0, self.f_pad),
        #                                        bias=False)
        self.deconv = nn.Conv2d(inter_channel, 2, k_size, stride=(1, 1), padding=(0, self.f_pad), bias=False)

    def forward(self, x, dec_conv_add_cache):
        x_with_cache = torch.cat([dec_conv_add_cache, x], dim=2)
        dec_conv_add_cache = x[..., -self.t_pad:, :]
        x = self.deconv(x_with_cache)  # [B, 2, T, F]
        return x, dec_conv_add_cache


class StreamPureTransformerPSE(nn.Module):
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
        self.input_conv = StreamCausalConv2d(2, enc_conv_channels)
        self.encoder = StreamTimeAttn(attn_dim=attn_dim,
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
        self.up_conv = StreamCausalConv2d(enc_conv_channels, fusion_dim)
        self.fusion_block = FusionModule(embedding_dim, fusion_dim, time_chunk_size, dropout=dropout, drop_path=drop_path)
        self.down_conv = StreamCausalConv2d(fusion_dim, dec_conv_channels)

        # ____________DECODER_________________
        self.decoder = StreamTimeAttn(attn_dim=attn_dim,
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

    def forward(self, x, emb, inp_conv_cache, enc_attn_cache_k, enc_attn_cache_v, up_conv_cache, down_conv_cache,
                dec_attn_cache_k, dec_attn_cache_v):
        embedding = F.normalize(emb, p=2, dim=-1)
        src_signal = x
        x, inp_conv_cache = self.input_conv(x, inp_conv_cache)
        x, enc_attn_cache_k, enc_attn_cache_v = self.encoder(x, enc_attn_cache_k, enc_attn_cache_v)
        x, up_conv_cache = self.up_conv(x, up_conv_cache)
        x = self.fusion_block(x, embedding)
        x, down_conv_cache = self.down_conv(x, down_conv_cache)
        x, dec_attn_cache_k, dec_attn_cache_v = self.decoder(x, dec_attn_cache_k, dec_attn_cache_v)
        x = self.crm(x, src_signal)

        return x, inp_conv_cache, enc_attn_cache_k, enc_attn_cache_v, up_conv_cache, down_conv_cache, dec_attn_cache_k, dec_attn_cache_v


if __name__ == "__main__":

    inter_d_model = 384
    num_heads = 12
    emb_dim = 256
    freq_size = 161
    freq_patch_size = 23
    time_chunk_size = 8
    time_patch_size = 2
    layer_num = 1
    left_context_chunk_num = 2
    enc_layer_num = 8
    dec_layer_num = 9
    enc_conv_channels = 16
    dec_conv_channels = 24
    fusion_dim = 128

    stream_model = StreamPureTransformerPSE(attn_dim=inter_d_model, num_heads=num_heads,
                                            enc_conv_channels=enc_conv_channels, dec_conv_channels=dec_conv_channels,
                                            fusion_dim=fusion_dim,
                                            embedding_dim=emb_dim, time_chunk_size=time_chunk_size, time_patch_size=time_patch_size,
                                            left_context_chunk_number=left_context_chunk_num, enc_layer_num=enc_layer_num,
                                            dec_layer_num=dec_layer_num, dropout=0.2,
                                            drop_path=0.15, freq_dim=freq_size).eval()

    bs = 1
    x = torch.randn(bs, 2, 400, freq_size)
    print("INPUT SHAPE: ", x.shape)
    emb = torch.rand(bs, emb_dim)

    head_dim = inter_d_model // num_heads
    left_context = (freq_size // freq_patch_size) * (time_chunk_size // time_patch_size) * left_context_chunk_num

    inp_conv_cache = torch.zeros(bs, 2, 2, freq_size)
    enc_attn_cache_k = torch.zeros(enc_layer_num, num_heads, head_dim, left_context)
    enc_attn_cache_v = torch.zeros(enc_layer_num, num_heads, left_context, head_dim)
    up_conv_cache = torch.zeros(bs, enc_conv_channels, 2, freq_size)
    down_conv_cache = torch.zeros(bs, fusion_dim, 2, freq_size)
    dec_attn_cache_k = torch.zeros(dec_layer_num, num_heads, head_dim, left_context)
    dec_attn_cache_v = torch.zeros(dec_layer_num, num_heads, left_context, head_dim)

    stream_outputs = []
    for i in range(0, x.shape[2], time_chunk_size):
        chunk = x[..., i:i + time_chunk_size, :]
        (stream_res, inp_conv_cache, enc_attn_cache_k, enc_attn_cache_v,
         up_conv_cache, down_conv_cache, dec_attn_cache_k, dec_attn_cache_v) = stream_model(chunk, emb,
                                                                                            inp_conv_cache,
                                                                                            enc_attn_cache_k,
                                                                                            enc_attn_cache_v,
                                                                                            up_conv_cache,
                                                                                            down_conv_cache,
                                                                                            dec_attn_cache_k,
                                                                                            dec_attn_cache_v)
        stream_outputs.append(stream_res.detach())

    stream_output = torch.cat(stream_outputs, dim=2)
    print("OUTPUT SHAPE: ", stream_output.shape)


