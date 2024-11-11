import h5py

from .feature_map_operations import compute_pairwise_distances
from .util import get_relevant_layer_names, reeig_on_valueerror
from ..modules import BiMap, SCMPool, ReEig, CovariancePool


def do_lbl_dist_analysis(savedir, feature_maps):

    layers = [BiMap, SCMPool, ReEig, CovariancePool]  # layer that output SPD mats
    with h5py.File(savedir.joinpath('dists.h5'), 'w') as h:
        for layer_name in get_relevant_layer_names(layers=layers, feature_maps=feature_maps):
            for metric in ['euclid', 'logeuclid', 'riemann']:

                try:
                    dists, needed_reeig = reeig_on_valueerror(
                        fn=compute_pairwise_distances,
                        layer_name=layer_name,
                        layer_names_ls=[SCMPool.__name__, CovariancePool.__name__,
                                        f'{SCMPool.__name__}_0', f'{CovariancePool.__name__}_0',
                                        f'{BiMap.__name__}', f'{BiMap.__name__}_0',
                                        ],
                        feature_map=feature_maps[layer_name],
                        metric=metric
                    )

                    h.create_dataset(f'{layer_name}/{metric}', data=dists, compression='gzip', compression_opts=9)
                    h.attrs[f'{layer_name}_{metric}_needed_reeig'] = needed_reeig

                except ValueError:
                    # assuming val err is a PD problem, hopefully from one metric only so skip
                    print(f'Still got a reeig, so skipping {layer_name=} with {metric=}')
                    continue
