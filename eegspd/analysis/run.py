import time
from functools import partial

from torch.utils.data import DataLoader, Subset

from .freq_gain import do_freq_gain_analysis
from .lbl_mdm import do_lbl_mdm_analysis
from .lbl_svm import do_lbl_svm_analysis
from .lbl_eigs import do_lbl_eig_analysis
from .lbl_dists import do_lbl_dist_analysis
from .cov_grads import do_cov_grad_analysis
from .util import get_feature_maps


def run_analyses(savedir, model, train_set, test_set, analyses_to_run, sfreq):

    # surely there's a nicer way to do this?
    for (X_train, y_train, _) in DataLoader(train_set, batch_size=len(train_set), shuffle=False, drop_last=False):
        break
    for (X_test, y_test, _) in DataLoader(test_set, batch_size=len(test_set), shuffle=False, drop_last=False):
        break

    train_feature_maps = get_feature_maps(model, X_train)
    test_feature_maps = get_feature_maps(model, X_test)

    if isinstance(train_set, Subset):
        labels_dict = train_set.dataset.datasets[0].window_kwargs[0][1]['mapping']
    else:
        labels_dict = train_set.datasets[0].window_kwargs[0][1]['mapping']


    analysis_fns = {
        'freq_gain': partial(
            do_freq_gain_analysis,
            savedir=savedir,
            X=X_train,
            feature_maps=train_feature_maps,
            fs=sfreq
        ),
        'lbl_mdm': partial(
            do_lbl_mdm_analysis,
            savedir=savedir,
            train_feature_maps=train_feature_maps,
            y_train=y_train,
            test_feature_maps=test_feature_maps,
            y_test=y_test
        ),
        'lbl_svm': partial(
            do_lbl_svm_analysis,
            savedir=savedir,
            train_feature_maps=train_feature_maps,
            y_train=y_train,
            test_feature_maps=test_feature_maps,
            y_test=y_test
        ),
        'lbl_eigs': partial(
            do_lbl_eig_analysis,
            savedir=savedir,
            feature_maps=train_feature_maps,
            mode='svd'
        ),
        'lbl_dists': partial(
            do_lbl_dist_analysis,
            savedir=savedir,
            feature_maps=train_feature_maps,
        ),
        'cov_grads_softmax': partial(
            do_cov_grad_analysis,
            savedir=savedir,
            model=model,
            X=X_train,
            y=y_train,
            labels_dict=labels_dict,
            softmax=True
        ),
    }

    for analysis in analyses_to_run:
        if model._get_name() == 'FBSPDNet':
            # The rest of the analysis exclusion is in the main fn
            if analysis in ['freq_gain', 'cov_grads_softmax']:
                print(f'skipping {analysis} because model is FBSPDNet')
                continue
        start = time.time()
        analysis_fns[analysis]()
        dur = time.time() - start
        print(f'Analysis: {analysis} took {dur / 60:.1f}mins')
