import torch
from reformer_pytorch import Reformer

model = Reformer(
    dim = 512,
    depth = 12,
    heads = 8,
    lsh_dropout = 0.1,
    causal = True
).cuda()

x = torch.randn(1, 8192, 512).cuda()
y = model(x) # (1, 8192, 512)
print(y)