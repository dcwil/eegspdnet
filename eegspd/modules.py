import math
from typing import Tuple

import numpy as np
import geoopt
import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.types import Number
import torch.nn as nn
import torch.nn.functional as F
from geoopt.tensor import ManifoldParameter
from geoopt.manifolds import Stiefel, Sphere
from einops import einsum, rearrange
from einops.layers.torch import Reduce, Rearrange

from . import functional

########## Taken from https://github.com/rkobler/TSMNet/blob/main/spdnets/modules.py

# TODO: add in interband removal (instead of rearranging in the conv layers, rearrange the deletions?)
# TODO: add in OAS regularisation
class CovariancePool(nn.Module):
    def __init__(self, alpha=None, unitvar=False):
        super().__init__()
        self.pooldim = -1
        self.chandim = -2
        self.alpha = alpha
        self.unitvar = unitvar

    def forward(self, X: Tensor) -> Tensor:
        X0 = X - X.mean(dim=self.pooldim, keepdim=True)
        if self.unitvar:
            X0 = X0 / X0.std(dim=self.pooldim, keepdim=True)
            X0.nan_to_num_(0)

        C = (X0 @ X0.transpose(-2, -1)) / X0.shape[self.pooldim]  # TODO: Doesn't this need a - 1?
        if self.alpha is not None:  # TODO: surely this doesn't change anything?
            Cd = C.diagonal(dim1=self.pooldim, dim2=self.pooldim - 1)
            Cd += self.alpha
        return C


# Used in teh DANN - do we need it?
class ReverseGradient(nn.Module):
    def __init__(self, scaling=1.):
        super().__init__()
        self.scaling_ = scaling

    def forward(self, X: Tensor) -> Tensor:
        return functional.reverse_gradient.apply(X, self.scaling_)


class BiMap(nn.Module):
    def __init__(self, shape: Tuple[int, ...] or torch.Size, W0: Tensor = None, manifold='stiefel', **kwargs):
        super().__init__()

        if manifold == 'stiefel':
            assert (shape[-2] >= shape[-1])
            mf = Stiefel()  # TODO: add option for canonical?
        elif manifold == 'sphere':
            mf = Sphere()
            shape = list(shape)
            shape[-1], shape[-2] = shape[-2], shape[-1]
        else:
            raise NotImplementedError()

        # add constraint (also initializes the parameter to fulfill the constraint)
        self.W = ManifoldParameter(torch.empty(shape, **kwargs), manifold=mf)

        # optionally initialize the weights (initialization has to fulfill the constraint!)
        if W0 is not None:
            self.W.data = W0  # e.g., self.W = torch.nn.init.orthogonal_(self.W)
        else:
            self.reset_parameters()

    def forward(self, X: Tensor) -> Tensor:
        if isinstance(self.W.manifold, Sphere):
            return self.W @ X @ self.W.transpose(-2, -1)
        else:
            return self.W.transpose(-2, -1) @ X @ self.W

    @torch.no_grad()
    def reset_parameters(self):
        if isinstance(self.W.manifold, Stiefel):
            # uniform initialization on stiefel manifold after theorem 2.2.1 in Chikuse (2003): statistics on special manifolds
            W = torch.rand(self.W.shape, dtype=self.W.dtype, device=self.W.device)
            self.W.data = W @ functional.sym_invsqrtm.apply(W.transpose(-1, -2) @ W)
        elif isinstance(self.W.manifold, Sphere):
            W = torch.empty(self.W.shape, dtype=self.W.dtype, device=self.W.device)
            # kaiming initialization std2uniformbound * gain * fan_in
            bound = math.sqrt(3) * 1. / W.shape[-1]
            W.uniform_(-bound, bound)
            # constraint has to be satisfied
            self.W.data = W / W.norm(dim=-1, keepdim=True)
        else:
            raise NotImplementedError()


class ReEig(nn.Module):
    # NOTE: edited the typical 1e-4 threshold due to some computation issues
    def __init__(self, threshold: Number = 5e-4):  # TODO: where does the epsilon selection come into play?
        super().__init__()
        self.threshold = Tensor([threshold])

    def forward(self, X: Tensor) -> Tensor:
        return functional.sym_reeig.apply(X, self.threshold)


class LogEig(nn.Module):
    def __init__(self, ndim, tril=True, sqrt=True):
        super().__init__()

        self.tril = tril
        self.sqrt = sqrt  # I added this toggle
        if self.tril:
            ixs_lower = torch.tril_indices(ndim, ndim, offset=-1)
            ixs_diag = torch.arange(start=0, end=ndim, dtype=torch.long)
            self.ixs = torch.cat((ixs_diag[None, :].tile((2, 1)), ixs_lower), dim=1)  # TODO: double check: why not use TrilEmbedder?
        self.ndim = ndim

    def forward(self, X: Tensor) -> Tensor:
        return self.embed(functional.sym_logm.apply(X))

    def embed(self, X: Tensor) -> Tensor:
        if self.tril:
            x_vec = X[..., self.ixs[0], self.ixs[1]]
            if self.sqrt:
                x_vec[..., self.ndim:] *= math.sqrt(2)
        else:
            x_vec = X.flatten(start_dim=-2)
        return x_vec


class TrilEmbedder(nn.Module):

    def forward(self, X: Tensor) -> Tensor:
        ndim = X.shape[-1]
        ixs_lower = torch.tril_indices(ndim, ndim, offset=-1)
        ixs_diag = torch.arange(start=0, end=ndim, dtype=torch.long)
        ixs = torch.cat((ixs_diag[None, :].tile((2, 1)), ixs_lower), dim=1)

        x_vec = X[..., ixs[0], ixs[1]]
        x_vec[..., ndim:] *= math.sqrt(2)
        return x_vec

    def inverse_transform(self, x_vec: Tensor) -> Tensor:
        ndim = int(-.5 + math.sqrt(.25 + 2 * x_vec.shape[-1]))  # c*(c+1)/2 = nts
        ixs_lower = torch.tril_indices(ndim, ndim, offset=-1)
        ixs_diag = torch.arange(start=0, end=ndim, dtype=torch.long)

        X = torch.zeros(x_vec.shape[:-1] + (ndim, ndim), device=x_vec.device, dtype=x_vec.dtype)

        # off diagonal elements
        X[..., ixs_lower[0], ixs_lower[1]] = x_vec[..., ndim:] / math.sqrt(2)
        X[..., ixs_lower[1], ixs_lower[0]] = x_vec[..., ndim:] / math.sqrt(2)
        X[..., ixs_diag, ixs_diag] = x_vec[..., :ndim]

        return X

###### My Stuff


class ChSpecConv(nn.Module):

    def __init__(self, n_elecs, n_filters, filter_time_length):
        super(__class__, self).__init__()
        self.conv = nn.Conv1d(
            n_elecs, n_filters * n_elecs, filter_time_length, groups=n_elecs
        )

    def forward(self, X):
        X = self.conv(X)
        return X


class SpatConv(nn.Module):
    def __init__(self, n_in_chs, n_out_chs):
        super(__class__, self).__init__()
        self.conv = nn.Sequential(
            Rearrange('b e t -> b 1 t e'),
            nn.Conv2d(1, n_out_chs, (1, n_in_chs)),
            Rearrange('b e_f t e -> b (e e_f) t'),
        )

    def forward(self, X):
        X = self.conv(X)
        return X


class ChIndConv(nn.Module):
    def __init__(self, n_filters, filter_time_length):
        super().__init__()

        self.conv = nn.Sequential(
            Rearrange('b e t -> b 1 t e'),
            nn.Conv2d(1, n_filters, (filter_time_length, 1)),
            Rearrange('b e_f t e -> b (e e_f) t'),  # this results in electrodes staying next to one another
        )

    def forward(self, X):
        return self.conv(X)


class RemoveInterbandCovariance(nn.Module):
    # ch_list should contain channel labels post conv ie [0, 0, 0, 1, 1, 1] or [0, 1, 0, 1, 0, 1]
    def __init__(self, elecs_grouped, n_filters, n_elecs, keep_inter_elec=False):
        super().__init__()
        self.keep_inter_elec = keep_inter_elec
        m = functional.create_interband_mask(
            elecs_grouped=elecs_grouped,
            n_filters=n_filters,
            n_elecs=n_elecs,
            keep_inter_elec=keep_inter_elec
        )
        self.register_buffer('mask', m)

    def forward(self, x):
        x = einsum(x, self.mask, 'b c1 c2, c1 c2 -> b c1 c2')
        return x


class ChIndSincConv(nn.Module):
    """Adapted from https://github.com/Popgun-Labs/SincNetConv"""

    def __init__(self, n_filters, filter_length, fs, spacing='rand', f_min=1, width_min=1, bias=None):
        super().__init__()

        self.fs = fs
        self.f_min = f_min
        self.width_min = width_min
        self._bias = bias

        self.nyquist_f = int((fs - 1) / 2)
        self.low_max = self.nyquist_f - self.width_min

        if spacing == 'mel':
            cutoffs = np.linspace(f_min, functional.hz2mel(self.nyquist_f), n_filters + 1)
            cutoffs = functional.mel2hz(cutoffs)
        elif spacing == 'hz':
            cutoffs = np.linspace(f_min, self.nyquist_f, n_filters + 1)
        elif spacing == 'rand':
            cutoffs = np.sort(np.random.uniform(low=f_min, high=self.nyquist_f + 1, size=n_filters + 1))

        # learnable params
        self.lows_ = nn.Parameter(torch.from_numpy(cutoffs[:-1] / fs).float())

        if spacing == 'rand':
            widths = np.random.uniform(low=1, high=self.nyquist_f + 1, size=n_filters)
        else:
            widths = np.diff(cutoffs)

        self.widths_ = nn.Parameter(torch.from_numpy(widths / fs).float())
        # windower
        window = torch.from_numpy(np.hamming(filter_length)).float()
        self.register_buffer('window', window)

        t_right = torch.linspace(1, (filter_length - 1) / 2, steps=int((filter_length - 1) / 2)) / fs
        self.register_buffer('t_right', t_right.float())

        if self._bias:
            tmp = nn.Conv1d(1, n_filters, filter_length)
            self.bias = nn.Parameter(tmp.bias.clone())  # double check with robin
        else:
            self.bias = None

    def forward(self, x):
        lows = torch.clamp(self.lows_, min=self.f_min / self.fs, max=self.low_max / self.fs)
        highs = lows + torch.clamp(self.widths_, min=self.width_min / self.fs)
        # highs = torch.clamp(highs, max=self.nyquist_f / self.fs)
        self.fb = torch.stack([lows * self.fs, highs * self.fs])

        # make a bandpass by subtracting two low pass filters from each other
        low_lowpass_fb = 2 * lows.view(-1, 1) * functional.sinc(lows * self.fs, self.t_right)
        high_lowpass_fb = 2 * highs.view(-1, 1) * functional.sinc(highs * self.fs, self.t_right)
        bandpass_fb = high_lowpass_fb - low_lowpass_fb

        # re-norm to 0-1
        band_maxs, _ = torch.max(bandpass_fb, dim=1, keepdim=True)
        bandpass_fb = bandpass_fb / band_maxs

        # window filters
        filters = bandpass_fb * self.window.view(1, -1)

        # apply all filters across all channels
        x = rearrange(x, 'b e t -> b 1 t e')
        filters = rearrange(filters, 'f1 f2 -> f1 1 f2 1')
        x = F.conv2d(x, filters, bias=self.bias)
        x = rearrange(x, 'b e_f t e -> b (e e_f) t')  # this results in electrodes staying next to one another
        return x


class ChSpecSincConv(nn.Module):
    """Adapted from https://github.com/Popgun-Labs/SincNetConv"""

    def __init__(self, n_elecs, n_filters, filter_length, fs, spacing='rand', f_min=1, width_min=1, bias=None):
        super().__init__()

        self.fs = fs
        self.f_min = f_min
        self.width_min = width_min
        self.n_ch = n_elecs
        self.n_filters = n_filters
        self._bias = bias

        self.nyquist_f = int((fs - 1) / 2)
        self.low_max = self.nyquist_f - self.width_min

        if spacing == 'mel':
            cutoffs = np.stack([np.linspace(f_min, functional.hz2mel(self.nyquist_f), n_filters + 1) for _ in range(n_elecs)])
            cutoffs = functional.mel2hz(cutoffs)
        elif spacing == 'hz':
            cutoffs = np.stack([np.linspace(f_min, self.nyquist_f, n_filters + 1) for _ in range(n_elecs)])
        elif spacing == 'rand':
            cutoffs = np.stack(
                [np.sort(np.random.uniform(low=f_min, high=self.nyquist_f + 1, size=n_filters + 1)) for _ in range(n_elecs)])

        # learnable params
        lows_ = np.hstack([x[:-1] for x in cutoffs])
        self.lows_ = nn.Parameter(torch.from_numpy(lows_ / fs).float())

        if spacing == 'rand':
            self.widths_ = nn.Parameter(torch.from_numpy(np.random.uniform(low=1, high=self.nyquist_f, size=len(self.lows_)) / fs).float())
        else:
            self.widths_ = nn.Parameter(torch.from_numpy(np.diff(cutoffs).flatten() / fs).float())

        # windower
        window = torch.from_numpy(np.hamming(filter_length)).float()
        self.register_buffer('window', window)

        t_right = torch.linspace(1, (filter_length - 1) / 2, steps=int((filter_length - 1) / 2)) / fs
        self.register_buffer('t_right', t_right.float())

        self.fb = None

        if self._bias:
            tmp = nn.Conv1d(n_elecs, n_filters * n_elecs, filter_length, groups=n_elecs)
            self.bias = nn.Parameter(tmp.bias.clone())
        else:
            self.bias = None

    def forward(self, x):
        lows = torch.clamp(self.lows_, min=self.f_min / self.fs, max=self.low_max / self.fs)
        highs = lows + torch.clamp(self.widths_, min=self.width_min / self.fs)
        # highs = torch.clamp(highs, max=self.nyquist_f / self.fs)

        self.fb = torch.stack([lows * self.fs, highs * self.fs])

        # make a bandpass by subtracting two low pass filters from each other
        low_lowpass_fb = 2 * lows.view(-1, 1) * functional.sinc(lows * self.fs, self.t_right)
        high_lowpass_fb = 2 * highs.view(-1, 1) * functional.sinc(highs * self.fs, self.t_right)
        bandpass_fb = high_lowpass_fb - low_lowpass_fb

        # re-norm to 0-1
        band_maxs, _ = torch.max(bandpass_fb, dim=1, keepdim=True)
        bandpass_fb = bandpass_fb / band_maxs

        # window filters
        filters = bandpass_fb * self.window.view(1, -1)

        # apply all filters across all channels
        x = F.conv1d(x, filters.unsqueeze(1), groups=self.n_ch, bias=self.bias)

        return x


class SPDDropout(nn.Module):

    def __init__(self, drop_prob=0.5, use_scaling=True, epsilon=1e-5):
        super().__init__()
        self.drop_prob = drop_prob
        self.use_scaling = use_scaling
        self.epsilon = epsilon

    def forward(self, x):

        if self.training:

            # batched so loop - VECTORISE???
            batch_size, _, _ = x.shape
            out = torch.zeros_like(x)

            for b_i in range(batch_size):
                out[b_i, :, :] = functional.dropout_spd(x[b_i, :, :], drop_prob=self.drop_prob,
                                             use_scaling=self.use_scaling, epsilon=self.epsilon)

            return out
        else:
            return x

# TODO: add AIRM
# TODO: allow for single lambda for whole batch
# TODO: add loss calculation helper fn
class SPDMixUp(nn.Module):

    def __init__(self, alpha=1.0, use_cuda=False):
        super().__init__()
        self.alpha = alpha
        self.use_cuda = use_cuda

    def forward(self, x):
        batch_size, _, _ = x.shape

        # generate mixing coeffs
        if self.alpha > 0:
            lams = [np.random.beta(self.alpha, self.alpha) for _ in range(batch_size)]
        else:
            lams = [1 for _ in range(batch_size)]

        lams = torch.from_numpy(np.array(lams))
        # generate shuffled idxs (whos mixing with who)
        if self.use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)

        x_a, x_b = x, x[index]
        mixed_x = functional.geodesic_logeuclid(x_a, x_b, alpha=lams)
        return mixed_x, lams, index


class SCMPool(nn.Module):
    def __init__(self, demean=False):
        super().__init__()
        self.demean = demean

    def forward(self, x: Tensor) -> Tensor:
        n_t = x.shape[-1]

        if self.demean:
            x = x - x.mean(dim=-1, keepdim=True)  # assume batch x chs x timepoints

        C = einsum(x, x, 'b c1 t, b c2 t -> b c1 c2')
        C = C / (n_t - 1)

        return C


class StiefelAdam:
    # copied from adavoudi, uses euclidean stiefel ops

    def __init__(self, optimizer, set_zero=True):
        self.optimizer = optimizer
        self.state = {}
        self.set_zero = set_zero

    def zero_grad(self):
        return self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    @staticmethod
    def _orthogonal_projection(x, u):
        # U - X sym(Xt) U
        return u - x @ geoopt.linalg.sym(x.transpose(-1, -2) @ u)

    @staticmethod
    def _retraction(x, u):
        q, r = torch.linalg.qr(x + u)
        unflip = geoopt.linalg.extract_diag(r).sign().add(0.5).sign()
        q *= unflip[..., None, :]
        return q

    def step(self, closure=None):

        for group in self.optimizer.param_groups:

            for point in group["params"]:
                grad = point.grad

                if grad is None:
                    continue

                if isinstance(point, ManifoldParameter):
                    if id(point) not in self.state:
                        self.state[id(point)] = point.data.clone()
                    else:
                        self.state[id(point)].fill_(0).add_(point.data.clone())

                    # geopt wd adds wd*point to grad here - before proj
                    # adavoudi had fill 0s on the point here
                    egrad2rgrad = self._orthogonal_projection(grad.data, point.data)
                    grad.data.fill_(0).add_(egrad2rgrad)

                    # I put a set zero here instead!
                    if self.set_zero:
                        point.data.fill_(0)

        loss = self.optimizer.step(closure)

        for group in self.optimizer.param_groups:
            for point in group['params']:

                grad = point.grad

                if grad is None:
                    continue

                if isinstance(point, ManifoldParameter):
                    trans = self._retraction(point.data, self.state[id(point)])
                    point.data.fill_(0).add_(trans)

        return loss

