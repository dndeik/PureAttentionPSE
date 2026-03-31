import torch

from model.full_model.full_model import PureTransformerPSE
from model.stream_model.stream_model import StreamPureTransformerPSE
from model.utils import count_parameters

# Init models
# The size of the chunk determines the number of time samples that the model will use in one step.
# In the case of a Fourier (n_fft=window_size=320, hop=160) and samplerate 16000 chunk 8 = 80 milliseconds

inter_d_model = 384
num_heads = 12
emb_dim = 256
freq_size = 161
freq_patch_size = 23
time_chunk_size = 8
time_patch_size = 2
layer_num = 1
left_context_chunk_num = 3
enc_layer_num = 8
dec_layer_num = 9
enc_conv_channels = 16
dec_conv_channels = 24
fusion_dim = 128

model = PureTransformerPSE(attn_dim=inter_d_model,
                           num_heads=num_heads,
                           enc_conv_channels=enc_conv_channels,
                           dec_conv_channels=dec_conv_channels,
                           fusion_dim=fusion_dim,
                           embedding_dim=emb_dim,
                           time_chunk_size=time_chunk_size,
                           time_patch_size=time_patch_size,
                           freq_dim=freq_size,
                           left_context_chunk_number=left_context_chunk_num,
                           enc_layer_num=enc_layer_num,
                           dec_layer_num=dec_layer_num,
                           dropout=0.2,
                           drop_path=0.15).eval()

stream_model = StreamPureTransformerPSE(attn_dim=inter_d_model,
                                        num_heads=num_heads,
                                        enc_conv_channels=enc_conv_channels,
                                        dec_conv_channels=dec_conv_channels,
                                        fusion_dim=fusion_dim,
                                        embedding_dim=emb_dim,
                                        time_chunk_size=time_chunk_size,
                                        time_patch_size=time_patch_size,
                                        freq_dim=freq_size,
                                        left_context_chunk_number=left_context_chunk_num,
                                        enc_layer_num=enc_layer_num,
                                        dec_layer_num=dec_layer_num,
                                        dropout=0.2,
                                        drop_path=0.15).eval()

stream_model.load_state_dict(model.state_dict())
count_parameters(stream_model)

# Init caches
# All caches init for following input data: Fourier transform with n_fft=window_size=320, hop=160
bs = 1
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

# Init dummy input (Fourier with n_fft=window_size=320, hop=160)
# Time should be dividable by chunk_size for correct comparison
x = torch.randn(bs, 2, 400, freq_size)

# Infer full model
full_res = model(x, emb)

# Infer stream model chunk by chunk
stream_chunks = []

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
    stream_chunks.append(stream_res.detach())

stream_res = torch.cat(stream_chunks, dim=2)

print("Models results are equal: ", torch.allclose(full_res, stream_res, atol=1e-4))
print("INPUT shape: ", x.shape)
print("RESULT shape: ", stream_res.shape)
