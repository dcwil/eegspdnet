import os
import json
import time
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score, StratifiedKFold

from .filter import apply_fb
from .pipelines import build_pipeline


def create_search_space(n_filters: int, n_elecs: int, fb_type: str):
    fb_type = fb_type.lower()
    assert fb_type in ['spec', 'ind']

    n_chans = n_filters * n_elecs if fb_type == 'spec' else n_filters

    search_space = {}
    for i in range(n_chans):
        search_space[f'low_{i}'] = (0, 0.5)
        search_space[f'width_{i}'] = (0, 0.5)

    return search_space


def _parse_bayes_opt_res(res):
    return {'mean_acc': res['target'], **res['params']}


def _check_last_iter(last_iter, n, skip_failed_iter, skip_value=-1):
    iter_failed = last_iter == skip_value

    if iter_failed and skip_failed_iter:
        print('Skipping failed iter...')
        return n
    elif iter_failed and not skip_failed_iter:
        print('Iter failed, not skipping...')

    return n - 1


def search(savepath: str | os.PathLike, objective: callable, search_space: dict, num_samples: int,
           seed: int, n_init_k: int = 10, time_budget_s: int = 43200, search_alg='bayes',
           skip_failed_iter: bool = True):
    savepath = Path(savepath)

    if search_alg == 'bayes':

        optim = BayesianOptimization(
            f=objective,
            pbounds=search_space,
            random_state=seed,
            allow_duplicate_points=True,
        )

        # split num samples into n_init and n_iter
        n_init = num_samples // n_init_k
        n_iter = num_samples - n_init
        print(f'Will search with {n_init=} and {n_iter=}')

        start = time.time()
        while (dur := (time.time() - start)) < time_budget_s:
            if n_init > 0:
                # run single init
                optim.maximize(init_points=1, n_iter=0)

                n_init = _check_last_iter(
                    last_iter=optim.res[-1]['target'],
                    n=n_init,
                    skip_failed_iter=not skip_failed_iter  # don't skip inital expore iters
                )

            elif n_iter > 0:
                # run single iter
                optim.maximize(init_points=0, n_iter=1)
                n_iter = _check_last_iter(
                    last_iter=optim.res[-1]['target'],
                    n=n_iter,
                    skip_failed_iter=skip_failed_iter
                )

            else:
                print(f'Finished all iter with {dur=}')
                break

        if n_init + n_iter > 0:
            print(f'Ran out of time with {n_init=}, {n_iter=} remaining')

        history = pd.DataFrame([_parse_bayes_opt_res(res) for res in optim.res])
        history.to_csv(savepath.joinpath('history.csv'))

        best = _parse_bayes_opt_res(optim.max)
        with open(savepath.joinpath('best.json'), 'w') as h:
            json.dump(best, h)

    else:
        raise ValueError


def objective(config, pipe, X, y, fb_params):
    # wrap this in a parse func
    n_params = len([x for x, _ in config.items() if x.startswith('low')])

    lows = [config[f'low_{i}'] for i in range(n_params)]
    widths = [config[f'width_{i}'] for i in range(n_params)]

    lows = torch.Tensor(lows)
    widths = torch.Tensor(widths)

    X_fb, _ = apply_fb(
        X=X,
        lows=lows,
        widths=widths,
        **fb_params
    )

    # do some nan checks
    if torch.isnan(X_fb).sum() > 0:
        print('Filtered array contains NaN!\nReturning -1 ...')
        return -1

    try:
        accs = cross_val_score(pipe,
                               X=X_fb.clone().detach().numpy(),
                               y=y.clone().detach().numpy(),
                               cv=StratifiedKFold(shuffle=True, n_splits=3)
                               )

        if any([np.isnan(v) for v in accs]):
            # nan accs can result from the FB creating no PD matrices
            raise ValueError('One of the CV folds contained a NaN!')

        out = accs.mean()
        # print(out, fb)
        return out
    except ValueError:
        print('Returning -1 score due to ValueError...')
        return -1


def run(
        savedir: os.PathLike | str,
        train_set: torch.utils.data.Dataset,
        n_elecs: int,
        n_filters: int,
        sfreq: int,
        seed: int,
        num_samples: int,
        time_budget_s: int = 43200,
        search_alg: str = 'bayes',
        cov_estimator: str = "scm",
        classifier: str = "mdm",
        metric: str = "logeuclid",
        remove_interband: bool = False,
        keep_inter_elec: int = None,
        fb_type: str = 'ind',
        spacing: str = 'rand',
):
    for (X_train, y_train, _) in DataLoader(train_set, batch_size=len(train_set), shuffle=False, drop_last=False):
        break

    search_space = create_search_space(
        n_filters=n_filters,
        n_elecs=n_elecs,
        fb_type=fb_type
    )
    pipe = build_pipeline(
        cov_estimator=cov_estimator,
        classifier=classifier,
        metric=metric,
        remove_interband=remove_interband,
        elecs_grouped=True,  # should alway sbe true for the sinc layers
        n_filters=n_filters,
        n_elecs=n_elecs,
        keep_inter_elec=keep_inter_elec,
    )

    fb_params = dict(
        n_filters=n_filters,
        fb_type=fb_type,
        fs=sfreq,
        n_elecs=n_elecs,
        spacing=spacing,
        bias=False,
        filter_length=25,
        f_min=1,
        width_min=1
    )

    def _objective(**kwargs):
        return objective(config=kwargs, pipe=pipe, X=X_train, y=y_train, fb_params=fb_params)

    # start search
    print('starting search...')
    search(
        savepath=savedir,
        seed=seed,
        objective=_objective,
        search_space=search_space,
        num_samples=num_samples,
        time_budget_s=time_budget_s,
        search_alg=search_alg,
    )
