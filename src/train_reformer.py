import torch
from reformer_pytorch import Reformer


def main():
    model = Reformer(
        dim=512,
        depth=12,
        heads=8,
        lsh_dropout=0.1,
        causal=True
    )  # .cuda()

    x = torch.randn(1, 8192, 512)  # .cuda()
    y = model(x)  # (1, 8192, 512)
    print(y)


if __name__ == '__main__':
    main()


def instantiate_reformer(input_channels=0, num_classes=1000):
    return Reformer(
        dim=input_channels,
        depth=12,
        heads=8,
        lsh_dropout=0.1,
        causal=True
    )  # .cuda()
