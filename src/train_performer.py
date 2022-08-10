import torch
from performer_pytorch import Performer


def main():
    model = Performer(
        dim=512,
        depth=1,
        heads=8,
        causal=True
    )

    x = torch.randn(1, 2048, 512)
    print(model(x))  # (1, 2048, 512)


if __name__ == "__main__":
    main()


def instantiate_performer(input_channels=0, num_classes=1000):
    return Performer(
        dim=input_channels,
        depth=1,
        heads=8,
        causal=True
    )
