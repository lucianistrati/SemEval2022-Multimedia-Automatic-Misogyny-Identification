import torch
from performer_pytorch import Performer

model = Performer(
    dim = 512,
    depth = 1,
    heads = 8,
    causal = True
)

x = torch.randn(1, 2048, 512)
print(model(x)) # (1, 2048, 512)
