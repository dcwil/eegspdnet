import torch
import h5py
import numpy as np
from scipy.interpolate import interp1d

from .feature_map_operations import compute_spectrogram
from .util import get_relevant_layer_names
from ..modules import ChSpecConv, ChIndConv, ChSpecSincConv, ChIndSincConv


def _get_gain_in_db(sig_in, sig_out):
    return 20 * torch.log10(sig_out / sig_in)


def calculate_freq_gain(pre_conv_feature_map, post_conv_feature_map, fs):  # TODO: change to fs
    # compute freq gain array for entire batch
    assert len(pre_conv_feature_map.shape) == 3  # b c t

    pre_conv_sp, pre_conv_freqs = compute_spectrogram(pre_conv_feature_map, fs)
    post_conv_sp, post_conv_freqs = compute_spectrogram(post_conv_feature_map, fs)

    # discard phase
    pre_conv_sp, post_conv_sp = pre_conv_sp.abs(), post_conv_sp.abs()

    N_elec = pre_conv_feature_map.shape[1]
    N_chs = post_conv_feature_map.shape[1]
    N_convs_per_elec = int(N_chs/N_elec)

    # Both ChInd and ChSpec duplicate channels when N_filters > 1
    # When this happens the duplicated channels are placed next to each other
    # ie. N_filters = 2 means that chs 0 and 1 are from elec 0
    # to match the shape of the input, we need to repeat along the electrode axis

    pre_conv_sp = pre_conv_sp.repeat_interleave(N_convs_per_elec, dim=1)  # b c t

    # now N chs is the same, but t on post_conv will be smaller (bc of the convolution)
    # to solve this we interpolate
    post_conv_sp_interp = torch.zeros_like(pre_conv_sp)
    for b_i, batch in enumerate(post_conv_sp):
        for sp_i, sp in enumerate(batch):
            interp_fn = interp1d(
                x=post_conv_freqs,
                y=sp,
                kind='cubic',
                fill_value='extrapolate'
            )

            # interp spectrogram to pre conv freq values
            sp_interp = interp_fn(x=pre_conv_freqs)
            post_conv_sp_interp[b_i, sp_i, :] = torch.from_numpy(sp_interp)

    # convert to power
    pre_conv_power = pre_conv_sp.pow(2)
    post_conv_power = post_conv_sp_interp.pow(2)

    # can now easily get the gain due to convolution
    gain = _get_gain_in_db(pre_conv_power, post_conv_power)
    return gain, pre_conv_freqs


def do_freq_gain_analysis(savedir, X, feature_maps, fs, avg_across_trials=True):
    layers = [ChIndConv, ChSpecConv, ChIndSincConv, ChSpecSincConv, torch.nn.Conv1d]
    for layer_name in get_relevant_layer_names(layers=layers, feature_maps=feature_maps):
        print(f'Doing frequency gain analysis for layer {layer_name}')
        gain, freqs = calculate_freq_gain(
            pre_conv_feature_map=X,
            post_conv_feature_map=feature_maps[layer_name],
            fs=fs
        )
        if avg_across_trials:
            print('avergaing frequency gain across trials....')
            gain = gain.numpy().mean(axis=0)
        else:
            gain = gain.numpy().astype(np.float16)
        with h5py.File(savedir.joinpath(f'freq_gain.h5'), 'w') as h:
            h.create_dataset('gain', data=gain, compression='gzip', compression_opts=9)
            h.attrs['freqs'] = freqs.tolist()
            h.attrs['layer_name'] = layer_name
            h.attrs['avg_across_trials'] = avg_across_trials

        print('Saved!')

