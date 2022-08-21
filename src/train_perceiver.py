from typing import Dict, List, Set, Tuple, Optional, Any, Callable, NoReturn, Union, Mapping, Sequence, Iterable
from perceiver_pytorch import Perceiver

import pdb


def instantiate_perceiver(input_channels: int = None, input_axis: int = None, num_classes: int = None):
    """

    :param input_channels:
    :param input_axis:
    :param num_classes:
    :return:
    """
    model = Perceiver(
        input_channels=input_channels or 1,  # number of channels for each token of the input
        input_axis=input_axis or 1,  # number of axis for input data (2 for images, 3 for video)
        num_freq_bands=2,  # number of freq bands, with original value (2 * K + 1)
        max_freq=10.,  # maximum frequency, hyperparameter depending on how fine the data is
        depth=2,  # depth of net. The shape of the final attention mechanism will be:
        #   depth * (cross attention -> self_per_cross_attn * self attention)
        num_latents=2,  # number of latents, or induced set points, or centroids. different papers giving it different names
        latent_dim=4,  # latent dimension
        cross_heads=1,  # number of heads for cross attention. paper said 1
        latent_heads=2,  # number of heads for latent self attention, 8
        cross_dim_head=4,  # number of dimensions per cross attention head
        latent_dim_head=4,  # number of dimensions per latent self attention head
        num_classes=num_classes or 1000,
        attn_dropout=0.,
        ff_dropout=0.,
        weight_tie_layers=False,  # whether to weight tie layers (optional, as indicated in the diagram)
        fourier_encode_data=True,  # whether to auto-fourier encode the data, using the input_axis given. defaults to
        # True, but can be turned off if you are fourier encoding the data yourself
        self_per_cross_attn=2  # number of self attention blocks per cross attention
    )
    return model
