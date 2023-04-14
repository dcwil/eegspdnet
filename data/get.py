from collections import OrderedDict

import numpy as np

from mne.filter import filter_data
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    preprocess,
    Preprocessor,
    filterbank,
    create_windows_from_events,
)


from .classes import EEGDataset


def _default_prepro(channels_to_pick, max_abs_val, s_freq, n_jobs, **kwargs):
    def _scale(x):
        return x * 1e6

    def _clip(x):
        return np.clip(x, -max_abs_val, max_abs_val)

    cfg = OrderedDict()
    cfg["scale"] = {"apply_on_array": True, "fn": _scale}
    cfg["pick_chs"] = {
        "apply_on_array": False,
        "fn": "pick_channels",
        "ordered": True,
        "ch_names": channels_to_pick,
    }
    cfg["clip"] = {"apply_on_array": True, "fn": _clip}
    cfg["resample"] = {
        "apply_on_array": False,
        "fn": "resample",
        "sfreq": s_freq,
        "verbose": False,
        "n_jobs": n_jobs,
    }

    return cfg


def _get_splits(
    subject, channels_to_pick, max_abs_val, s_freq, n_jobs=1, name="Schirrmeister2017"
):
    preprocessors_dict = _default_prepro(
        channels_to_pick=list(channels_to_pick),
        max_abs_val=max_abs_val,
        s_freq=s_freq,
        n_jobs=n_jobs,
    )
    preprocessors = [Preprocessor(**conf) for conf in preprocessors_dict.values()]
    ds = MOABBDataset(dataset_name=name, subject_ids=[subject])
    preprocess(concat_ds=ds, preprocessors=preprocessors)
    return ds


def _get_band_windows(
    split_ds,
    frequency_bands,
    channels_to_pick,
    trial_start_offset_samples,
    trial_stop_offset_samples,
    n_jobs=1,
    **kwargs,
):
    """For filtering the split dataset into each band. Args passed to Kwargs are ignored."""
    band_windows = []
    # Start doing each band
    if len(frequency_bands) > 1:
        raise ValueError(
            "If you want to do multiband filtering, use `_get_ch_spec_band_windows`"
        )
    for band_idx, band in enumerate(frequency_bands):
        # print(f'Doing band {band_idx}: {band}')
        fbank = [
            # starting from the second band, discard the old band from the
            # previous iteration by again picking original channels
            Preprocessor(
                apply_on_array=False,
                fn="pick_channels",
                ch_names=channels_to_pick,
                ordered=True,
            ),
            Preprocessor(
                apply_on_array=False,
                fn=filterbank,
                frequency_bands=[band],
                # instead of making a deep copy of the entire dataset, just do
                # not drop the original signals
                drop_original_signals=False,
                verbose=False,
                n_jobs=n_jobs,
            ),
        ]

        preprocess(concat_ds=split_ds, preprocessors=fbank)

        windows_ds = create_windows_from_events(
            concat_ds=split_ds,
            trial_start_offset_samples=trial_start_offset_samples,
            trial_stop_offset_samples=trial_stop_offset_samples,
            drop_last_window=False,
            n_jobs=n_jobs,
        )

        assert len(windows_ds.datasets) == 1  # should only be one?

        windows_d = windows_ds.datasets[0]
        # get labels
        labels = windows_d.windows.metadata["target"].to_numpy()

        # this is not nice, it should be possible to pick the first / last
        # half of signals instead. check braindecode PR:
        # https://github.com/braindecode/braindecode/pull/185/files
        chs = [ch for ch in windows_d.windows.ch_names if "_" in ch]

        # double check that channel ordering has not changed
        assert all([ch_.startswith(ch) for ch, ch_ in zip(channels_to_pick, chs)])

        windows = windows_d.windows.get_data(picks=chs)
        drop_log = windows_d.windows.drop_log

        band_windows.append((windows, labels, drop_log))

    return band_windows


def _ch_spec_get_band_windows(
    split_ds,
    frequency_bands,
    channels_to_pick,
    trial_start_offset_samples,
    trial_stop_offset_samples,
    s_freq,
    n_jobs=1,
    **kwargs,
):
    """frequency_bands should be a list of tuples, one per filter per electrode, of length n_bands * n_electrodes"""

    assert len(frequency_bands) % len(channels_to_pick) == 0
    n_filters = int(len(frequency_bands) / len(channels_to_pick))
    pick = [
        Preprocessor(
            apply_on_array=False,
            fn="pick_channels",
            ch_names=channels_to_pick,
            ordered=True,
        )
    ]
    preprocess(split_ds, preprocessors=pick)

    unfilt = split_ds.datasets[0].raw.copy()
    dupes = []
    for i in range(2, n_filters + 1):
        ch_mapping = {ch: f"{ch}_{i}" for ch in channels_to_pick}
        dupes.append(split_ds.datasets[0].raw.copy().rename_channels(ch_mapping))

    unfilt.add_channels(dupes)
    del dupes

    # filter channels individually
    filt = np.zeros_like(unfilt._data)
    for ch_i, sig in enumerate(unfilt._data):
        l_freq, h_freq = frequency_bands[ch_i]
        filt[ch_i] = filter_data(sig, sfreq=s_freq, l_freq=l_freq, h_freq=h_freq)

    # swap for filtered data
    split_ds.datasets[0].raw = unfilt
    del unfilt
    split_ds.datasets[0].raw._data = filt
    del filt

    windows_ds = create_windows_from_events(
        concat_ds=split_ds,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=trial_stop_offset_samples,
        drop_last_window=False,
        n_jobs=n_jobs,
    )
    windows_d = windows_ds.datasets[0]
    # get labels
    labels = windows_d.windows.metadata["target"].to_numpy()

    windows = windows_d.windows.get_data()
    drop_log = windows_d.windows.drop_log
    return [(windows, labels, drop_log)]


def _detect_set_name_on_split_ds(split_ds, name):
    if name == "Schirrmeister2017":
        set_name = split_ds.description["run"][0]
    elif name == "BNCI2014001":
        if "T" in split_ds.description["session"].values[0]:
            set_name = "train"
        else:
            set_name = "test"
    else:
        raise ValueError("dataset_name not recognised")

    return set_name


def get_data(
    sub, frequency_bands=None, ch_spec=False, name="Schirrmeister2017", **defaults
):
    freq_range = (defaults["low_cut_hz"], defaults["hi_cut_hz"])

    ds = _get_splits(
        subject=sub,
        channels_to_pick=tuple(defaults["channels_to_pick"]),
        max_abs_val=defaults["max_abs_val"],
        s_freq=defaults["s_freq"],
        name=name,
    )

    splits = ds.split([[i] for i in range(len(ds.datasets))])

    frequency_bands = [freq_range] if frequency_bands is None else frequency_bands

    data = {}
    for i, split_ds in splits.items():
        set_name = _detect_set_name_on_split_ds(split_ds, name)
        # print(f'doing {set_name}')

        if set_name not in data:
            data[set_name] = {}

        if ch_spec:
            band_windows = _ch_spec_get_band_windows(
                split_ds, frequency_bands=frequency_bands, **defaults
            )
        else:
            band_windows = _get_band_windows(
                split_ds, frequency_bands=frequency_bands, **defaults
            )

        # this won't be ram efficient
        for k in ["trials", "labels"]:
            if k not in data[set_name]:
                data[set_name][k] = []

        data[set_name]["trials"].append(band_windows[0][0])
        data[set_name]["labels"].append(band_windows[0][1])

    for set_name in ["train", "test"]:
        for key in ["trials", "labels"]:
            data[set_name][key] = np.concatenate(data[set_name][key])

    train = EEGDataset(
        features=data["train"]["trials"],
        labels=data["train"]["labels"],
        set_name="train",
    )

    test = EEGDataset(
        features=data["test"]["trials"],
        labels=data["test"]["labels"],
        set_name="test",
    )
    return train, test
