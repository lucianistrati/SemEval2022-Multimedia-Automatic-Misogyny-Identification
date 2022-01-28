import torch
from linear_attention_transformer import LinearAttentionTransformer

model = LinearAttentionTransformer(
    dim = 512,
    heads = 8,
    depth = 1,
    max_seq_len = 8192,
    n_local_attn_heads = 4
).cuda()

x = torch.randn(1, 8192, 512).cuda()
print(model(x)) # (1, 8192, 512)
