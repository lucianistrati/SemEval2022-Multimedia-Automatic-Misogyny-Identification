import torch
from sinkhorn_transformer import SinkhornTransformerLM
from sinkhorn_transformer import SinkhornTransformer

model = SinkhornTransformer(
    dim = 1024,
    heads = 8,
    depth = 12,
    bucket_size = 128
)

x = torch.randn(1, 2048, 1024)
model(x) # (1, 2048, 1024)
