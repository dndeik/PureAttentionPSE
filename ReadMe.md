# Pure Attention Personalized Speech Enhancement model (PureAttention-PSE)
 
## Overview
This repository presents a model for realtime streaming personalized speech enhancement in a TF-domain.
The model consists of four parts:
- Transformer encoder for patched TF-sequence (cross-freq) with 2D positional encoding 
- Fusion block with cross attention between audio features and speaker embedding (embedding should be produced by separated model)
- Transformer decoder, same as encoder
- CRM module for adding the original signal and the predicted mask

## Performance and Quality
The model works with chunks in Fourier space and is pretty fast.

**Number of model parameters: ~38.9M**

Fourier Parameters: n_fft = 320, hop = 160

| Chunk Size                  | xRTF |
|-----------------------------|------|
| 8 (80 ms) with default init | ~6.7 |

All measurements were made with models converted to **OpenVINO** with **Intel(R) Core(TM) i5-9600 CPU @ 3.10GHz**
using 6 thread

The [STFT-wrapper](https://github.com/dndeik/torch-stft-wrapper) was used for model. 

# Stream and Full results comparison
The model was originally designed to work in chunk-by-chunk streaming mode.
But for train or in case of short audio you should use full model.
Also, you can check that it produces identical results with the stream version if you run the `compare_stream_and_full.py`.

To convert to the stream version, you just need to transfer the state dict from the full model.

## TODO
- [ ] Add trained weights
- [ ] Add code to convert model to ONNX/OV format