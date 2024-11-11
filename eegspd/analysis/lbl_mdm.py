import pandas as pd
from pyriemann.classification import MDM
from torch.linalg import LinAlgError

from .feature_map_operations import classify_feature_maps
from .util import get_relevant_layer_names, reeig_on_valueerror
from ..modules import BiMap, SCMPool, ReEig, CovariancePool


def do_lbl_mdm_analysis(savedir, train_feature_maps, y_train, test_feature_maps, y_test):

    metrics = ['euclid', 'logeuclid', 'riemann']
    layers = [BiMap, SCMPool, ReEig, CovariancePool]  # layer that output SPD mats

    rows_ls = []
    for layer_name in get_relevant_layer_names(layers=layers, feature_maps=train_feature_maps):
        for metric in metrics:
            clf = MDM(metric=metric)

            try:
                res, needed_reeig = reeig_on_valueerror(
                    fn=classify_feature_maps,
                    layer_name=layer_name,
                    layer_names_ls=[SCMPool.__name__, CovariancePool.__name__,
                                    f'{SCMPool.__name__}_0', f'{CovariancePool.__name__}_0',
                                    f'{BiMap.__name__}', f'{BiMap.__name__}_0',
                                    ],
                    clf=clf,
                    feature_map_train=train_feature_maps[layer_name],
                    feature_map_test=test_feature_maps[layer_name],
                    labels_train=y_train,
                    labels_test=y_test
                )

                rows_ls.append(dict(
                    layer_name=layer_name,
                    metric=metric,
                    needed_reeig=needed_reeig,
                    **res
                ))
            except ValueError:
                # assuming val err is a PD problem, hopefully from one metric only so skip
                print(f'Still got a non-pd, so skipping {layer_name=} with {metric=}')
                continue
            except LinAlgError:
                print('torch raised linalg error, likely failed eig/svd convergence!')
                continue

    df = pd.DataFrame(rows_ls)
    df.to_csv(savedir.joinpath('LBL_MDM.csv'))
