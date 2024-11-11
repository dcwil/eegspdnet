import math
import torch
import numpy as np
from typing import Callable, Tuple
from typing import Any
from torch.autograd import Function, gradcheck
from torch.functional import Tensor
from torch.types import Number


########## Taken from https://github.com/rkobler/TSMNet/blob/main/spdnets/functionals.py

# define the epsilon precision depending on the tensor datatype
EPS = {torch.float32: 1e-4, torch.float64: 1e-7}


def ensure_sym(A: Tensor) -> Tensor:
    """Ensures that the last two dimensions of the tensor are symmetric.
    Parameters
    ----------
    A : torch.Tensor
        with the last two dimensions being identical
    -------
    Returns : torch.Tensor
    """
    return 0.5 * (A + A.transpose(-1,-2))


def broadcast_dims(A: torch.Size, B: torch.Size, raise_error:bool=True) -> Tuple:
    """Return the dimensions that can be broadcasted.
    Parameters
    ----------
    A : torch.Size
        shape of first tensor
    B : torch.Size
        shape of second tensor
    raise_error : bool (=True)
        flag that indicates if an error should be raised if A and B cannot be broadcasted
    -------
    Returns : torch.Tensor
    """
    # check if the tensors can be broadcasted
    if raise_error:
        if len(A) != len(B):
            raise ValueError('The number of dimensions must be equal!')

    tdim = torch.tensor((A, B), dtype=torch.int32)

    # find differing dimensions
    bdims = tuple(torch.where(tdim[0].ne(tdim[1]))[0].tolist())

    # check if one of the different dimensions has size 1
    if raise_error:
        if not tdim[:,bdims].eq(1).any(dim=0).all():
            raise ValueError('Broadcast not possible! One of the dimensions must be 1.')

    return bdims


def sum_bcastdims(A: Tensor, shape_out: torch.Size) -> Tensor:
    """Returns a tensor whose values along the broadcast dimensions are summed.
    Parameters
    ----------
    A : torch.Tensor
        tensor that should be modified
    shape_out : torch.Size
        desired shape of the tensor after aggregation
    -------
    Returns : the aggregated tensor with the desired shape
    """
    bdims = broadcast_dims(A.shape, shape_out)

    if len(bdims) == 0:
        return A
    else:
        return A.sum(dim=bdims, keepdim=True)


def randn_sym(shape, **kwargs):
    ndim = shape[-1]
    X = torch.randn(shape, **kwargs)
    ixs = torch.tril_indices(ndim,ndim, offset=-1)
    X[...,ixs[0],ixs[1]] /= math.sqrt(2)
    X[...,ixs[1],ixs[0]] = X[...,ixs[0],ixs[1]]
    return X

# this is AIRM, I think
# TODO: adapt to handle different metrics
def spd_2point_interpolation(A : Tensor, B : Tensor, t : Number) -> Tensor:
    rm_sq, rm_invsq = sym_invsqrtm2.apply(A)
    return rm_sq @ sym_powm.apply(rm_invsq @ B @ rm_invsq, torch.tensor(t)) @ rm_sq


class reverse_gradient(Function):
    """
    Reversal of the gradient
    Parameters
    ---------
    scaling : Number
        A constant number that is multiplied to the sign-reversed gradients (1.0 default)
    """
    @staticmethod
    def forward(ctx, x, scaling = 1.0):
        ctx.scaling = scaling
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.scaling
        return grad_output, None


class sym_modeig:
    """Basic class that modifies the eigenvalues with an arbitrary elementwise function
    """

    @staticmethod
    def forward(M : Tensor, fun : Callable[[Tensor], Tensor], fun_param : Tensor = None,
                ensure_symmetric : bool = False, ensure_psd : bool = False) -> Tensor:
        """Modifies the eigenvalues of a batch of symmetric matrices in the tensor M (last two dimensions).

        Source: Brooks et al. 2019, Riemannian batch normalization for SPD neural networks, NeurIPS

        Parameters
        ----------
        M : torch.Tensor
            (batch) of symmetric matrices
        fun : Callable[[Tensor], Tensor]
            elementwise function
        ensure_symmetric : bool = False (optional)
            if ensure_symmetric=True, then M is symmetrized
        ensure_psd : bool = False (optional)
            if ensure_psd=True, then the eigenvalues are clamped so that they are > 0
        -------
        Returns : torch.Tensor with modified eigenvalues
        """
        if ensure_symmetric:
            M = ensure_sym(M)

        # compute the eigenvalues and vectors
        s, U = torch.linalg.eigh(M)
        if ensure_psd:
            s = s.clamp(min=EPS[s.dtype])

        # modify the eigenvalues
        smod = fun(s, fun_param)
        X = U @ torch.diag_embed(smod) @ U.transpose(-1,-2)

        return X, s, smod, U

    @staticmethod
    def backward(dX : Tensor, s : Tensor, smod : Tensor, U : Tensor,
                 fun_der : Callable[[Tensor], Tensor], fun_der_param : Tensor = None) -> Tensor:
        """Backpropagates the derivatives

        Source: Brooks et al. 2019, Riemannian batch normalization for SPD neural networks, NeurIPS

        Parameters
        ----------
        dX : torch.Tensor
            (batch) derivatives that should be backpropagated
        s : torch.Tensor
            eigenvalues of the original input
        smod : torch.Tensor
            modified eigenvalues
        U : torch.Tensor
            eigenvector of the input
        fun_der : Callable[[Tensor], Tensor]
            elementwise function derivative
        -------
        Returns : torch.Tensor containing the backpropagated derivatives
        """

        # compute Lowener matrix
        # denominator
        L_den = s[...,None] - s[...,None].transpose(-1,-2)
        # find cases (similar or different eigenvalues, via threshold)
        is_eq = L_den.abs() < EPS[s.dtype]
        L_den[is_eq] = 1.0
        # case: sigma_i != sigma_j
        L_num_ne = smod[...,None] - smod[...,None].transpose(-1,-2)
        L_num_ne[is_eq] = 0
        # case: sigma_i == sigma_j
        sder = fun_der(s, fun_der_param)
        L_num_eq = 0.5 * (sder[...,None] + sder[...,None].transpose(-1,-2))
        L_num_eq[~is_eq] = 0
        # compose Loewner matrix
        L = (L_num_ne + L_num_eq) / L_den
        dM = U @  (L * (U.transpose(-1,-2) @ ensure_sym(dX) @ U)) @ U.transpose(-1,-2)
        return dM


class sym_reeig(Function):
    """
    Rectifies the eigenvalues of a batch of symmetric matrices in the tensor M (last two dimensions).
    """
    @staticmethod
    def value(s : Tensor, threshold : Tensor) -> Tensor:
        return s.clamp(min=threshold.item())

    @staticmethod
    def derivative(s : Tensor, threshold : Tensor) -> Tensor:
        return (s>threshold.item()).type(s.dtype)

    @staticmethod
    def forward(ctx: Any, M: Tensor, threshold : Tensor, ensure_symmetric : bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_reeig.value, threshold, ensure_symmetric=ensure_symmetric)
        ctx.save_for_backward(s, smod, U, threshold)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U, threshold = ctx.saved_tensors
        return sym_modeig.backward(dX, s, smod, U, sym_reeig.derivative, threshold), None, None

    @staticmethod
    def tests():
        """
        Basic unit tests and test to check gradients
        """
        ndim = 2
        nb = 1
        # generate random base SPD matrix
        A = torch.randn((1,ndim,ndim), dtype=torch.double)
        U, s, _ = torch.linalg.svd(A)

        threshold = torch.tensor([1e-3], dtype=torch.double)

        # generate batches
        # linear case (all eigenvalues are above the threshold)
        s = threshold * 1e1 + torch.rand((nb,ndim), dtype=torch.double) * threshold
        M = U @ torch.diag_embed(s) @ U.transpose(-1,-2)

        assert (sym_reeig.apply(M, threshold, False).allclose(M))
        M.requires_grad_(True)
        assert(gradcheck(sym_reeig.apply, (M, threshold, True)))

        # non-linear case (some eigenvalues are below the threshold)
        s = torch.rand((nb,ndim), dtype=torch.double) * threshold
        s[::2] += threshold
        M = U @ torch.diag_embed(s) @ U.transpose(-1,-2)
        assert (~sym_reeig.apply(M, threshold, False).allclose(M))
        M.requires_grad_(True)
        assert(gradcheck(sym_reeig.apply, (M, threshold, True)))

        # linear case, all eigenvalues are identical
        s = torch.ones((nb,ndim), dtype=torch.double)
        M = U @ torch.diag_embed(s) @ U.transpose(-1,-2)
        assert (sym_reeig.apply(M, threshold, True).allclose(M))
        M.requires_grad_(True)
        assert(gradcheck(sym_reeig.apply, (M, threshold, True)))


class sym_abseig(Function):
    """
    Computes the absolute values of all eigenvalues for a batch symmetric matrices.
    """
    @staticmethod
    def value(s : Tensor, param:Tensor = None) -> Tensor:
        return s.abs()

    @staticmethod
    def derivative(s : Tensor, param:Tensor = None) -> Tensor:
        return s.sign()

    @staticmethod
    def forward(ctx: Any, M: Tensor, ensure_symmetric : bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_abseig.value, ensure_symmetric=ensure_symmetric)
        ctx.save_for_backward(s, smod, U)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U = ctx.saved_tensors
        return sym_modeig.backward(dX, s, smod, U, sym_abseig.derivative), None


class sym_logm(Function):
    """
    Computes the matrix logarithm for a batch of SPD matrices.
    Ensures that the input matrices are SPD by clamping eigenvalues.
    During backprop, the update along the clamped eigenvalues is zeroed
    """
    @staticmethod
    def value(s : Tensor, param:Tensor = None) -> Tensor:
        # ensure that the eigenvalues are positive
        return s.clamp(min=EPS[s.dtype]).log()

    @staticmethod
    def derivative(s : Tensor, param:Tensor = None) -> Tensor:
        # compute derivative
        sder = s.reciprocal()
        # pick subgradient 0 for clamped eigenvalues
        sder[s<=EPS[s.dtype]] = 0
        return sder

    @staticmethod
    def forward(ctx: Any, M: Tensor, ensure_symmetric : bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_logm.value, ensure_symmetric=ensure_symmetric)
        ctx.save_for_backward(s, smod, U)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U = ctx.saved_tensors
        return sym_modeig.backward(dX, s, smod, U, sym_logm.derivative), None


class sym_expm(Function):
    """
    Computes the matrix exponential for a batch of symmetric matrices.
    """
    @staticmethod
    def value(s : Tensor, param:Tensor = None) -> Tensor:
        return s.exp()

    @staticmethod
    def derivative(s : Tensor, param:Tensor = None) -> Tensor:
        return s.exp()

    @staticmethod
    def forward(ctx: Any, M: Tensor, ensure_symmetric : bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_expm.value, ensure_symmetric=ensure_symmetric)
        ctx.save_for_backward(s, smod, U)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U = ctx.saved_tensors
        return sym_modeig.backward(dX, s, smod, U, sym_expm.derivative), None


class sym_powm(Function):
    """
    Computes the matrix power for a batch of symmetric matrices.
    """
    @staticmethod
    def value(s : Tensor, exponent : Tensor) -> Tensor:
        return s.pow(exponent=exponent)

    @staticmethod
    def derivative(s : Tensor, exponent : Tensor) -> Tensor:
        return exponent * s.pow(exponent=exponent-1.)

    @staticmethod
    def forward(ctx: Any, M: Tensor, exponent : Tensor, ensure_symmetric : bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_powm.value, exponent, ensure_symmetric=ensure_symmetric)
        ctx.save_for_backward(s, smod, U, exponent)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U, exponent = ctx.saved_tensors
        dM = sym_modeig.backward(dX, s, smod, U, sym_powm.derivative, exponent)

        dXs = (U.transpose(-1,-2) @ ensure_sym(dX) @ U).diagonal(dim1=-1,dim2=-2)
        dexp = dXs * smod * s.log()

        return dM, dexp, None


class sym_sqrtm(Function):
    """
    Computes the matrix square root for a batch of SPD matrices.
    """
    @staticmethod
    def value(s : Tensor, param:Tensor = None) -> Tensor:
        return s.clamp(min=EPS[s.dtype]).sqrt()

    @staticmethod
    def derivative(s : Tensor, param:Tensor = None) -> Tensor:
        sder = 0.5 * s.rsqrt()
        # pick subgradient 0 for clamped eigenvalues
        sder[s<=EPS[s.dtype]] = 0
        return sder

    @staticmethod
    def forward(ctx: Any, M: Tensor, ensure_symmetric : bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_sqrtm.value, ensure_symmetric=ensure_symmetric)
        ctx.save_for_backward(s, smod, U)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U = ctx.saved_tensors
        return sym_modeig.backward(dX, s, smod, U, sym_sqrtm.derivative), None


class sym_invsqrtm(Function):
    """
    Computes the inverse matrix square root for a batch of SPD matrices.
    """
    @staticmethod
    def value(s : Tensor, param:Tensor = None) -> Tensor:
        return s.clamp(min=EPS[s.dtype]).rsqrt()

    @staticmethod
    def derivative(s : Tensor, param:Tensor = None) -> Tensor:
        sder = -0.5 * s.pow(-1.5)
        # pick subgradient 0 for clamped eigenvalues
        sder[s<=EPS[s.dtype]] = 0
        return sder

    @staticmethod
    def forward(ctx: Any, M: Tensor, ensure_symmetric : bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_invsqrtm.value, ensure_symmetric=ensure_symmetric)
        ctx.save_for_backward(s, smod, U)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U = ctx.saved_tensors
        return sym_modeig.backward(dX, s, smod, U, sym_invsqrtm.derivative), None


class sym_invsqrtm2(Function):
    """
    Computes the square root and inverse square root matrices for a batch of SPD matrices.
    """

    @staticmethod
    def forward(ctx: Any, M: Tensor, ensure_symmetric : bool = False) -> Tensor:
        Xsq, s, smod, U = sym_modeig.forward(M, sym_sqrtm.value, ensure_symmetric=ensure_symmetric)
        smod2 = sym_invsqrtm.value(s)
        Xinvsq = U @ torch.diag_embed(smod2) @ U.transpose(-1,-2)
        ctx.save_for_backward(s, smod, smod2, U)
        return Xsq, Xinvsq

    @staticmethod
    def backward(ctx: Any, dXsq: Tensor, dXinvsq: Tensor):
        s, smod, smod2, U = ctx.saved_tensors
        dMsq = sym_modeig.backward(dXsq, s, smod, U, sym_sqrtm.derivative)
        dMinvsq = sym_modeig.backward(dXinvsq, s, smod2, U, sym_invsqrtm.derivative)

        return dMsq + dMinvsq, None


class sym_invm(Function):
    """
    Computes the inverse matrices for a batch of SPD matrices.
    """
    @staticmethod
    def value(s : Tensor, param:Tensor = None) -> Tensor:
        return s.clamp(min=EPS[s.dtype]).reciprocal()

    @staticmethod
    def derivative(s : Tensor, param:Tensor = None) -> Tensor:
        sder = -1. * s.pow(-2)
        # pick subgradient 0 for clamped eigenvalues
        sder[s<=EPS[s.dtype]] = 0
        return sder

    @staticmethod
    def forward(ctx: Any, M: Tensor, ensure_symmetric : bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_invm.value, ensure_symmetric=ensure_symmetric)
        ctx.save_for_backward(s, smod, U)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U = ctx.saved_tensors
        return sym_modeig.backward(dX, s, smod, U, sym_invm.derivative), None


def spd_mean_kracher_flow(X : Tensor, G0 : Tensor = None, maxiter : int = 50, dim = 0, weights = None, return_dist = False, return_XT = False) -> Tensor:

    if X.shape[dim] == 1:
        if return_dist:
            return X, torch.tensor([0.0], dtype=X.dtype, device=X.device)
        else:
            return X

    if weights is None:
        n = X.shape[dim]
        weights = torch.ones((*X.shape[:-2], 1, 1), dtype=X.dtype, device=X.device)
        weights /= n

    if G0 is None:
        G = (X * weights).sum(dim=dim, keepdim=True)
    else:
        G = G0.clone()

    nu = 1.
    dist = tau = crit = torch.finfo(X.dtype).max
    i = 0

    while (crit > EPS[X.dtype]) and (i < maxiter) and (nu > EPS[X.dtype]):
        i += 1

        Gsq, Ginvsq = sym_invsqrtm2.apply(G)
        XT = sym_logm.apply(Ginvsq @ X @ Ginvsq)
        GT = (XT * weights).sum(dim=dim, keepdim=True)
        G = Gsq @ sym_expm.apply(nu * GT) @ Gsq

        if return_dist:
            dist = torch.norm(XT - GT, p='fro', dim=(-2,-1))
        crit = torch.norm(GT, p='fro', dim=(-2,-1)).max()
        h = nu * crit
        if h < tau:
            nu = 0.95 * nu
            tau = h
        else:
            nu = 0.5 * nu

    if return_dist:
        return G, dist
    if return_XT:
        return G, XT
    return G


######### Stuff not form KObler

def regularise_with_oas(matrix, n_samples, n_features):
    """Recreate regularise with oas func in pytorch"""
    trc = matrix.diagonal(offset=0, dim1=-1, dim2=-2).sum(
        -1
    )  # https://discuss.pytorch.org/t/get-the-trace-for-a-batch-of-matrices/108504
    mu = trc / n_features

    alpha = (
        (matrix**2)
            .view(matrix.shape[0], matrix.shape[1] * matrix.shape[1])
            .mean(axis=1)
    )
    num = alpha + mu**2
    den = (n_samples + 1.0) * (alpha - (mu**2) / n_features)
    shrinkage = torch.clip(num / den, max=1)
    shrunk_cov = (1.0 - shrinkage).view(matrix.shape[0], 1, 1) * matrix
    k = (shrinkage * mu).repeat_interleave(n_features).view(matrix.shape[0], n_features)
    shrunk_cov.diagonal(dim1=-2, dim2=-1)[
    :
    ] += k  # https://discuss.pytorch.org/t/operation-on-diagonals-of-matrix-batch/50779

    return shrunk_cov

def sinc(band, t_right):
    """
    :param band: (n_filt)
    :param t_right: K = filt_dim // 2 - 1
    :return: (n_filt, filt_dim)
    """
    n_filt = band.size(0)
    band = band[:, None]  # (n_filt, 1)
    t_right = t_right[None, :]  # (1, K)
    y_right = torch.sin(2 * math.pi * band * t_right) / (2 * math.pi * band * t_right)  # (n_filt, K)
    y_left = torch.flip(y_right, [1])
    y = torch.cat([y_left, torch.ones([n_filt, 1], device=band.device), y_right], dim=1)  # (n_filt, filt_dim)
    return y


def get_mel_points(fs, n_filt, fmin=80):
    """
    Return `n_filt` points linearly spaced in the mel-scale (in Hz)
    with an upper frequency of fs / 2
    :param fs: the sample rate in Hz
    :return: np.array (n_filt)
    """
    high_freq_mel = (2595 * np.log10(1 + (fs / 2) / 700))
    mel_points = np.linspace(fmin, high_freq_mel, n_filt)  # equally spaced in mel scale
    f_cos = (700 * (10 ** (mel_points / 2595) - 1))
    return f_cos


def get_bands(f_cos, fs):
    """
    :param f_cos: vector of mel-scaled frequency (n_filt)
    :param fs: audio sample rate
    :return (b1, b1)
        b1: vector of lower cutoffs (n_filt)
        b2: vector of upper cutoffs
    """
    b1 = np.roll(f_cos, 1)
    b2 = np.roll(f_cos, -1)
    b1[0] = 30
    b2[-1] = (fs / 2) - 100
    return b1, b2

def hz2mel(hz):
    return (2595 * np.log10(1 + (hz / 700)))

def mel2hz(mel):
    return (700 * (10 ** (mel / 2595) - 1))

# TODO: add in mean value of remaining diags for epsilon?
def dropout_spd(input_mat: torch.Tensor, drop_prob: float = 0.5, training: bool = True,
                inplace: bool = False, use_scaling: bool = True, epsilon=1) -> torch.Tensor:
    """Applies dropout to covariance matrix. When a channel is dropped the diagonal is set to epsilon and the off diagonal to 0. A drop
    prob of 1 means all neurons get dropped"""

    if drop_prob < 0.0 or drop_prob > 1.0:
        raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(drop_prob))
    if inplace:
        raise NotImplementedError("Inplace dropout is not supported")
    if not training:
        return input

    scaling = 1 / (1 - drop_prob)

    mask_flat = torch.empty(input_mat.shape[-1]).bernoulli_(1 - drop_prob)
    mask_flat = mask_flat.unsqueeze(0)  # make a 1, X matrix

    mask_offdiag_sq = mask_flat.t().mm(mask_flat)  # this gets us a mask that works except for the diagonal off neurons

    mask_flat_opp = torch.ones_like(mask_flat) - mask_flat  # this gets us the ids of the off neurons

    mask_flat_opp = mask_flat_opp / input_mat.diag()  # make diagonals 1/diag so the multiplying mask will  ID them
    mask_flat_opp = mask_flat_opp * epsilon  # this means multiplying by mask gives epsilon
    mask = torch.diag_embed(mask_flat_opp) + mask_offdiag_sq

    output = input_mat * mask

    if use_scaling:
        output = output * scaling

    return output

# adapted from pyriemann
def geodesic_logeuclid(A, B, alpha):
    """Log-Euclidean geodesic between SPD/HPD matrices.

    The matrix at position :math:`\alpha` on the Log-Euclidean geodesic
    between two SPD/HPD matrices :math:`\mathbf{A}` and :math:`\mathbf{B}` is:

    .. math::
        \mathbf{C} = \exp \left( (1-\alpha) \log(\mathbf{A})
                     + \alpha \log(\mathbf{B}) \right)

    :math:`\mathbf{C}` is equal to :math:`\mathbf{A}` if :math:`\alpha` = 0,
    and :math:`\mathbf{B}` if :math:`\alpha` = 1.

    Parameters
    ----------
    A : ndarray, shape (..., n, n)
        First SPD/HPD matrices.
    B : ndarray, shape (..., n, n)
        Second SPD/HPD matrices.
    alpha : float, default=0.5
        The position on the geodesic.

    Returns
    -------
    C : ndarray, shape (..., n, n)
        SPD/HPD matrices on the Log-Euclidean geodesic.
    """

    log_A = sym_logm.apply(A)
    log_B = sym_logm.apply(B)

    # TODO: update to latest
    log_interp = ((1 - alpha.unsqueeze(-1).unsqueeze(-1)) * log_A) + (alpha.unsqueeze(-1).unsqueeze(-1) * log_B)

    return sym_expm.apply(log_interp)


def sym2vec_n_unique(ndim):
    # helper for calculating unique elements in symmetric square matrix
    return int(0.5 * ndim * (ndim + 1))


def create_interband_mask(elecs_grouped: bool, n_filters: int, n_elecs: int, keep_inter_elec: bool = False):
    """`keep_inter_elec=True` will create a mask that removes interband covariance between the same
    electrodes only."""

    filters = torch.arange(n_filters)
    chs = torch.arange(n_elecs)

    filters_list = filters.repeat(n_elecs) if elecs_grouped else filters.repeat_interleave(n_elecs)
    ch_list = chs.repeat_interleave(n_filters) if elecs_grouped else chs.repeat(n_filters)  # could also do a variable swap?
    ch_idxs = torch.arange(len(ch_list))

    if keep_inter_elec:
        m1 = ch_list.unsqueeze(-1) != ch_list.unsqueeze(0)  # inter elec off diags
        m2 = ch_idxs.unsqueeze(-1) == ch_idxs.unsqueeze(0)  # diags
        m = m1 | m2
    else:
        m = filters_list.unsqueeze(-1) == filters_list.unsqueeze(0)

    return m