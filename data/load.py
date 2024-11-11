from functools import lru_cache

import numpy as np
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    Preprocessor,
    preprocess,
    create_windows_from_events,
    Filter,
)


@lru_cache(maxsize=2)
def load_windows_dataset_cached(tuple_k, tuple_v):
    d = dict(zip(tuple_k, tuple_v))
    return load_windows_dataset(**d)


def load_windows_dataset(
    dataset,
    subject_id,
    channels_to_pick,
    max_abs_val,
    sfreq,
    trial_start_offset_samples,
    trial_stop_offset_samples,
    scaling_factor,
    hi_cut_hz,
    low_cut_hz,
    **kwargs,
):
    print(f"Downloading {dataset} S{subject_id}...")

    if dataset == 'Shin2017A':
        dataset_kwargs = {'accept': True}
    else:
        dataset_kwargs = {}

    ds = MOABBDataset(dataset_name=dataset, subject_ids=[subject_id], dataset_kwargs=dataset_kwargs)

    print("Doing prepro...")
    pick_scale_clip_resample_filter(
        dataset=ds,
        channels_to_pick=channels_to_pick,
        scaling_factor=scaling_factor,
        max_abs_val=max_abs_val,
        sfreq=sfreq,
        h_freq=hi_cut_hz,
        l_freq=low_cut_hz,
    )

    # Create windows using braindecode function for this. It needs parameters to define how
    # trials should be used.
    windows_dataset = create_windows_from_events(
        ds,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=trial_stop_offset_samples,
        preload=True,
    )
    return windows_dataset


def pick_scale_clip_resample_filter(
    dataset, channels_to_pick, scaling_factor, max_abs_val, sfreq, l_freq, h_freq
):
    preprocessors = [
        Preprocessor(fn="pick_channels", ch_names=channels_to_pick),
        Preprocessor(fn=lambda data: np.multiply(data, scaling_factor)),
        Preprocessor(fn=lambda data: np.clip(data, -max_abs_val, max_abs_val)),
        Preprocessor("resample", sfreq=sfreq),
        Filter(l_freq=l_freq, h_freq=h_freq),
    ]
    preprocess(dataset, preprocessors)
