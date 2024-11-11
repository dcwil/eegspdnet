from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from pyriemann.estimation import Covariances
from pyriemann.classification import MDM
from pyriemann.tangentspace import TangentSpace

from .util import RemoveInterbandCovariance_sk, ReEig_sk


def build_pipeline(
    cov_estimator="scm",
    classifier="mdm",
    metric="logeuclid",
    remove_interband=False,
    reeig=True,
    elecs_grouped=None,
    n_filters=None,
    n_elecs=None,
    keep_inter_elec=None,
):


    pipeline_ls = list()

    # covariance
    pipeline_ls.append((cov_estimator, Covariances(estimator=cov_estimator)))

    if remove_interband:
        assert type(elecs_grouped) == bool
        assert type(keep_inter_elec) == bool
        assert type(n_filters) == int
        assert type(n_elecs) == int

        pipeline_ls.append(
            (
                'rmint',
                RemoveInterbandCovariance_sk(
                    elecs_grouped=elecs_grouped,
                    n_filters=n_filters,
                    n_elecs=n_elecs,
                    keep_inter_elec=keep_inter_elec,
                )
            )
        )

    if reeig:
        pipeline_ls.append(('reeig', ReEig_sk()))

    if classifier == "mdm":
        pipeline_ls.append((classifier, MDM(metric=metric)))
    elif classifier == "svm_linear":
        pipeline_ls.append(("ts", TangentSpace(metric=metric)))
        pipeline_ls.append((classifier, SVC(kernel='linear')))
    else:
        raise ValueError

    return Pipeline(pipeline_ls)
