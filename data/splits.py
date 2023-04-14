from torch.utils.data import DataLoader

from .get import EEGDataset


def load_splits(
    train,
    test,
    dataload_params=None,
    mode="eval",
    batch_size=256,
    shuffle=False,
    drop_last=False,
    return_dataloaders=True,
):
    if dataload_params is None:
        dataload_params = {
            "train": {
                "batch_size": batch_size,
                "drop_last": drop_last,
                "num_workers": 1,
                "shuffle": shuffle,
            },
            "test": {"batch_size": 64, "drop_last": False, "num_workers": 1},
        }

    else:
        print(
            "dataload_params will override batch_size, shuffle and drop_last, be aware!"
        )

    assert mode in ["valid", "eval"]

    if mode == "eval":
        if return_dataloaders:
            train = DataLoader(train, **dataload_params["train"])
            test = DataLoader(test, **dataload_params["test"])

        return train, test
    else:
        # assuming 80:20 split for valid
        del test

        N = train.features.shape[0]
        split_i = int(N * 0.8)

        new_train = EEGDataset(
            features=train.features[:split_i, :, :],
            labels=train.labels[:split_i],
            set_name="train",
        )

        new_test = EEGDataset(
            features=train.features[split_i:, :, :],
            labels=train.labels[split_i:],
            set_name="test",
        )

        if return_dataloaders:
            new_train = DataLoader(new_train, **dataload_params["train"])
            new_test = DataLoader(new_test, **dataload_params["test"])

        return new_train, new_test
