import torch
import numpy as np


def get_layer_nos(n_features, n_layers, vect=True):
    """Gets layer numbers assuming each layer halves feature dim."""
    layers = [int(n_features / (2**i)) for i in range(n_layers)]

    if vect:
        final_no = (layers[-1] / 2) * (layers[-1] + 1)
    else:
        final_no = layers[-1] ** 2

    layers.append(final_no)

    return [int(layer) for layer in layers]


def _get_band_indices(n_chans, n_filters):
    return torch.LongTensor(
        [
            [i for i in range(n_chans * n_filters) if i % n_filters == f]
            for f in range(n_filters)
        ]
    ).flatten()


def blockdiag_submatrix(idx: int or slice, diag_idx: int, sizes: list):
    """Helper func for indexing a specific submatrix in an array of block diagonal matrices
    :param idx: int, the index of the block diagonal matrix (assuming array of multiple block diagonal matrices). Set
    to slice(None) to select all.
    :param diag_idx: int, the index along the diagonal of the submatrix (usually band_idx)
    :param sizes: list, output of get_matrix_sizes, ordered list of submatrix sizes
    """

    assert diag_idx <= len(sizes)
    start = sum(sizes[:diag_idx])
    end = sum(sizes[: diag_idx + 1])
    return np.s_[idx, start:end, start:end]
