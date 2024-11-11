from functools import partial

import torch.nn as nn

from ..modules import ChSpecSincConv, ChIndSincConv


def apply_fb(
    X,
    lows,
    widths,
    fb_type="spec",
    n_filters=1,
    filter_length=25,
    fs=250,
    n_elecs=None,
    spacing="rand",
    bias=False,
    f_min=1,
    width_min=1,
):
    """Creates a sinc layer with lows and widths as params, returns filtered data and filter bank"""
    fb_type = fb_type.lower()
    assert fb_type in ["spec", "ind"]

    sinc_layer = (
        partial(ChSpecSincConv, n_elecs=n_elecs) if fb_type == "spec" else ChIndSincConv
    )

    # instantiate
    sinc_layer = sinc_layer(
        n_filters=n_filters,
        filter_length=filter_length,
        fs=fs,
        spacing=spacing,
        bias=bias,
        f_min=f_min,
        width_min=width_min,
    )

    assert (
        lows.shape == sinc_layer.lows_.shape
    ), f"{sinc_layer.lows_.shape=} != {lows.shape=}"
    assert (
        widths.shape == sinc_layer.widths_.shape
    ), f"{sinc_layer.widths_.shape=} != {widths.shape=}"

    sinc_layer.lows_ = nn.Parameter(lows)
    sinc_layer.widths_ = nn.Parameter(widths)

    # filter
    X = sinc_layer(X)
    return X, sinc_layer.fb
