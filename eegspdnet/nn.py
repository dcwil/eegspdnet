import torch
import torch.nn as nn
import numpy as np

from spdnet.spd import SPDVectorize
from braindecode.models.modules import Expression

from .functional import (
    _regularise_with_oas_pytorch,
    ReEig_I,
    ReEig_DK,
    LogEig_I,
    LogEig_DK,
)
from .util import _get_band_indices, blockdiag_submatrix


class SCMPool(nn.Module):
    def __init__(self, n_filters, remove_interband=True, add_d=True, regularise=False):
        super(__class__, self).__init__()
        self.n_filters = n_filters
        self.remove_interband = remove_interband
        self.add_d = add_d  # needed for torchspdnet
        self.regularise = regularise

    def forward(self, X):
        batch_size, n_features, n_samps = X.shape
        # get actual n_chans
        n_chans = n_features // self.n_filters

        scm = (1 / (n_samps - 1)) * X.matmul(X.transpose(-1, -2))

        if self.regularise:
            scm = _regularise_with_oas_pytorch(scm, n_samps, n_features)

        if self.remove_interband:
            # set off diags to zero - definitely a better way to do this
            sizes = [n_chans for _ in range(self.n_filters)]
            idx_temp = np.ones(scm.shape)
            for band_idx in range(self.n_filters):
                idx = blockdiag_submatrix(
                    idx=slice(None), diag_idx=band_idx, sizes=sizes
                )
                idx_temp[idx] = 0

            idxs_actual = np.where(idx_temp == 1)

            scm[idxs_actual] = 0

        if len(scm.shape) == 3 and self.add_d:
            return scm[:, None, :, :]
        return scm


class ChSpecConv(nn.Module):
    def __init__(self, matrix_size, n_filters, n_time_kernel):
        super(__class__, self).__init__()
        self.conv = nn.Conv1d(
            matrix_size, n_filters * matrix_size, n_time_kernel, groups=matrix_size
        )
        self.band_indices = _get_band_indices(matrix_size, n_filters)

    def forward(self, x):
        x = self.conv(x)
        # reshape into bands
        x = torch.index_select(x, dim=1, index=self.band_indices)
        return x


class ChIndConv(nn.Module):
    def __init__(self, n_filters, n_time_kernel):
        super(__class__, self).__init__()

        self.conv = nn.Sequential(
            Expression(lambda x: x.unsqueeze(-1).transpose(1, 3)),
            nn.Conv2d(1, n_filters, (n_time_kernel, 1)),
            Expression(
                lambda x: x.transpose(2, 3).reshape(
                    x.shape[0], x.shape[1] * x.shape[3], x.shape[2]
                )
            ),
        )

    def forward(self, x):
        return self.conv(x)


class AltLogEig(nn.Module):
    """Adapted from adavoudi/spdnet to include switching between backprops"""

    def __init__(self, input_size, vectorise=True, mode="I"):
        super(AltLogEig, self).__init__()
        self.vectorise = vectorise
        self.mode = mode
        if vectorise:
            self.vec = SPDVectorize(input_size)

    def forward(self, input):
        if self.mode == "I":
            log = LogEig_I
        elif self.mode == "DK":
            log = LogEig_DK
        else:
            raise ValueError

        output = log.apply(input)

        if self.vectorise:
            output = self.vec(output)

        return output


class AltReEig(nn.Module):
    def __init__(self, epsilon=1e-4, mode="I"):
        super(AltReEig, self).__init__()
        self.register_buffer("epsilon", torch.FloatTensor([epsilon]))
        self.mode = mode

    def forward(self, input):
        if self.mode == "I":
            re = ReEig_I
        elif self.mode == "DK":
            re = ReEig_DK
        else:
            raise ValueError

        output = re.apply(input, self.epsilon)
        return output
