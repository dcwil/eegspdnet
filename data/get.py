from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

from .load import load_windows_dataset, load_windows_dataset_cached


def get_dataset_from_cfg(cfg, cache=False, mode='valid'):

    if cache:
        cfg['channels_to_pick'] = tuple(cfg['channels_to_pick'])  # cant cache a list

        # can't cache a dict either so:
        _cfg = {k: v for k, v in cfg.items() if k not in ['split']}  # remove a k: v we don't need
        tuple_k = tuple(_cfg.keys())
        tuple_v = tuple(_cfg.values())
        print(f'{tuple_k=}')
        print(f'{tuple_v=}')
        windows_dataset = load_windows_dataset_cached(tuple_k, tuple_v)
    else:
        windows_dataset = load_windows_dataset(**cfg)

    print(f'{windows_dataset.description=}')

    split_params = cfg['split'][mode]  # this is a dataset specific dict that will split the braindecode dataset
    print(f'{split_params}')

    # not all datasets can be split train, val & test using .split() (ie HGD)
    # float for test set indicates subsetting of train
    if type(split_params['test']) == float:
        print(f'Taking {split_params["test"]} of train set for test set...')

        test_set_size = split_params.pop("test")
        split_dataset = windows_dataset.split(split_params)['train']

        train_indices, val_indices = train_test_split(
            range(len(split_dataset)), test_size=test_set_size, shuffle=False
        )
        train_set = Subset(split_dataset, train_indices)
        test_set = Subset(split_dataset, val_indices)

    elif type(split_params['test']) == list:
        split_dataset = windows_dataset.split(split_params)
        train_set, test_set = split_dataset['train'], split_dataset['test']
    else:
        raise ValueError('split params must be list of indices to split by, or test split must be '
                         'float (percentage) of train set to split')
    return train_set, test_set