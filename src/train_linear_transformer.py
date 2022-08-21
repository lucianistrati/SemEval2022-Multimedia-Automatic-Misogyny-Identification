from typing import Dict, List, Set, Tuple, Optional, Any, Callable, NoReturn, Union, Mapping, Sequence, Iterable
import torch
from linear_attention_transformer import LinearAttentionTransformer
import pdb


def main():
    model = LinearAttentionTransformer(
        dim=512,
        heads=8,
        depth=1,
        max_seq_len=8192,
        n_local_attn_heads=4
    )  # .cuda()

    x = torch.randn(1, 8192, 512)  # .cuda()
    print(model(x))  # (1, 8192, 512)


if __name__ == '__main__':
    main()


def instantiate_linear_attention_transformer(input_channels=0, num_classes=1000):
    return LinearAttentionTransformer(
        dim=512,
        heads=8,
        depth=1,
        max_seq_len=8192,
        n_local_attn_heads=4
    )  # .cuda()
