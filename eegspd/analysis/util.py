from collections import OrderedDict
from typing import OrderedDict as TOrderedDict
from typing import List

import torch
import torch.nn as nn

from ..modules import ReEig


def get_feature_maps(
        model: nn.Module,
        input_data: torch.Tensor,
        return_logits: bool = False,
        no_grad: bool = True,

) -> TOrderedDict[str, torch.Tensor]:
    feature_maps = OrderedDict()

    def hook_feature_map(module, input, output):
        i = 0
        while True:
            name = f'{module._get_name()}_{i}'
            if name not in feature_maps:
                break
            i += 1
        feature_maps[name] = output

    def register_hooks(model):
        hooks = []
        for layer in model.children():
            print(layer._get_name())
            if isinstance(layer, nn.Sequential):
                hooks.extend(register_hooks(layer))
            else:
                hook = layer.register_forward_hook(hook_feature_map)
                hooks.append(hook)
        return hooks

    hooks = register_hooks(model)

    if no_grad:
        model.eval()
        with torch.no_grad():
            logits = model(input_data)
    else:
        logits = model(input_data)

    for hook in hooks:
        hook.remove()

    if return_logits:
        return logits, feature_maps
    else:
        return feature_maps


def get_relevant_layer_names(
        layers: List[nn.Module],
        feature_maps: TOrderedDict[str, torch.Tensor]
) -> List[str]:

    out = []
    layer_names = [l.__name__ for l in layers]
    for layer_name, feature_map in feature_maps.items():
        layer_name_trunc = layer_name.split('_')[0]
        if layer_name_trunc in layer_names:
            out.append(layer_name)

    return out


def reeig_on_valueerror(fn, layer_name=None, layer_names_ls=None, **kwargs):
    """tries fn with given args, if a ValueError is raised, it tries again after applying a reeig to any feature maps.
    Also only applies on layers in the layer_names_ls
    """

    do_layers = False
    if layer_name is not None:
        assert type(layer_name) == str
        assert layer_names_ls is not None

    if layer_names_ls is not None:
        assert type(layer_names_ls) == list
        assert layer_name is not None
        do_layers = True

    needed_reeig = False

    try:
        out = fn(**kwargs)
    except ValueError as e:

        if do_layers:
            if layer_name not in layer_names_ls:
                print(f'ValueError raised for {layer_name=} not in {layer_names_ls=}')
                raise ValueError(e)

        print(f'Trying reeig with {layer_name=} after ValueError: {e}')
        re = ReEig()

        for k, v in kwargs.items():
            if k.startswith('feature_map'):
                print(f'applying reeig on {k}')
                kwargs[k] = re(v)

        try:
            # redo fn  post-reeig
            out = fn(**kwargs)
            needed_reeig = True
        except ValueError as e:
            print('post-reeig fn call still failed!!')
            print(e)
            print('Trying again with thresh 1e-3')
            re_ = ReEig(threshold=1e-3)
            for k, v in kwargs.items():
                if k.startswith('feature_map'):
                    kwargs[k] = re_(v)
            out = fn(**kwargs)
            needed_reeig = True

    return out, needed_reeig
