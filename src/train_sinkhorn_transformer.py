import torch
from sinkhorn_transformer import SinkhornTransformerLM
from sinkhorn_transformer import SinkhornTransformer


def main():
    model = SinkhornTransformer(
        dim=1024,
        heads=8,
        depth=12,
        bucket_size=128
    )

    x = torch.randn(1, 2048, 1024)
    model(x)  # (1, 2048, 1024)


if __name__ == "__main__":
    main()


def instantiate_sinkhorn_transformer(input_channels=0, num_classes=1000):
    return SinkhornTransformer(
        dim=input_channels,
        heads=8,
        depth=12,
        bucket_size=128
    )
