import inspect
from _warnings import warn
from collections import OrderedDict

import torch.nn as nn
from torch import nn as nn

import eegspd.models as models
from eegspd.functional import sym2vec_n_unique
from eegspd.modules import BiMap, SPDDropout, ReEig, LogEig

models_dict = {}

def _init_models_dict():
    for m in inspect.getmembers(models, inspect.isclass):
        if issubclass(m[1], nn.Module):
            models_dict[m[0]] = m[1]


def _parse_bimap_sizes(bimap_sizes: tuple, n_elecs: int, n_filters: int):
    if type(bimap_sizes) == tuple:
        assert n_elecs is not None
        assert n_filters is not None
        k, n_bimap_reeig = bimap_sizes
        print(f'Doing bimap sizes with {k=} and {n_bimap_reeig=}')
        bimap_sizes = [int((n_elecs * n_filters) / k ** i) for i in range(n_bimap_reeig + 1)]
    else:
        print('fixed bimap sizes were passed in')
    return bimap_sizes


def create_spdnet(bimap_sizes, n_elecs=None, n_filters=None, n_classes=None, threshold=5e-4, log=True, sqrt=True, dropout=0, dropout_scaling=True):

    bimap_sizes_ls = _parse_bimap_sizes(bimap_sizes, n_elecs=n_elecs, n_filters=n_filters)
    print(f'{bimap_sizes_ls=}')

    # check last layer smaller than n_classes
    if n_classes is not None:
        last_layer_size_flat = sym2vec_n_unique(ndim=bimap_sizes_ls[-1])
        if last_layer_size_flat < n_classes:
            warn('The given bimap sizes result in a last layer with fewer unique elements than there are classes!')

    # TODO: add canonical stiefel toggle
    n_bimap_reeig = len(bimap_sizes_ls) - 1

    layers = OrderedDict()

    for i in range(n_bimap_reeig):
        size_in, size_out = bimap_sizes_ls[i], bimap_sizes_ls[i + 1]
        print(f'adding bimap ({size_in}, {size_out})')
        layers[f'bimap{i}'] = BiMap(shape=(size_in, size_out))

        if dropout > 0:
            layers[f'spd_dropout{i}'] = SPDDropout(epsilon=threshold, use_scaling=dropout_scaling)

        layers[f'reeig{i}'] = ReEig(threshold=threshold)

    if log:
        layers['logeig'] = LogEig(ndim=size_out, sqrt=sqrt)

    spdnet = nn.Sequential(layers)
    return spdnet