import h5py

from .feature_map_operations import compute_eigs
from .util import get_relevant_layer_names, reeig_on_valueerror
from ..modules import BiMap, SCMPool, ReEig, CovariancePool


def do_lbl_eig_analysis(savedir, feature_maps, mode='eig'):

    layers = [BiMap, SCMPool, ReEig, CovariancePool]  # layer that output SPD mats
    with h5py.File(savedir.joinpath('lbl_eigs.h5'), 'w') as h:
        for layer_name in get_relevant_layer_names(layers=layers, feature_maps=feature_maps):

            try:
                eigs, needed_reeig = reeig_on_valueerror(
                    fn=compute_eigs,
                    layer_name=layer_name,
                    layer_names_ls=[SCMPool.__name__, CovariancePool.__name__,
                                    f'{SCMPool.__name__}_0', f'{CovariancePool.__name__}_0',
                                    f'{BiMap.__name__}', f'{BiMap.__name__}_0',
                                    ],
                    feature_map=feature_maps[layer_name],
                    mode=mode
                )
                h.attrs[f'{layer_name}_needed_reeig'] = needed_reeig
                h.create_dataset(layer_name, data=eigs, compression='gzip', compression_opts=9)
            except ValueError:
                # assuming val err is a PD problem, hopefully from one metric only so skip
                print(f'Still got a reeig, so skipping {layer_name=}')
                continue
        h.attrs['mode'] = mode
