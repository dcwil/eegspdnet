from builtins import NotImplementedError
from enum import Enum
from typing import Tuple, Union, Optional
import torch
from torch.functional import Tensor
import torch.nn as nn
from torch.types import Number

from geoopt.tensor import ManifoldParameter, ManifoldTensor
from geoopt.manifolds import Manifold

from . import functional as functionals


# IMPORTANT: broadly taken/adapted from  https://github.com/rkobler/TSMNet/blob/main/spdnets/batchnorm.py

class BatchNormTestStatsMode(Enum):
    BUFFER = 'buffer'
    REFIT = 'refit'
    ADAPT = 'adapt'
    

class BatchNormDispersion(Enum):
    NONE = 'mean'
    SCALAR = 'scalar'
    VECTOR = 'vector'
    

class BatchNormTestStatsInterface:
    def set_test_stats_mode(self, mode : BatchNormTestStatsMode):
        pass
    
    
class BaseBatchNorm(nn.Module, BatchNormTestStatsInterface):
    def __init__(self, eta = 1.0, eta_test = 0.1, test_stats_mode : BatchNormTestStatsMode = BatchNormTestStatsMode.BUFFER):
        super().__init__()
        self.eta = eta
        self.eta_test = eta_test
        self.test_stats_mode = test_stats_mode

    def set_test_stats_mode(self, mode : BatchNormTestStatsMode):
        self.test_stats_mode = mode


class SchedulableBatchNorm(BaseBatchNorm):
    def set_eta(self, eta = None, eta_test = None):
        if eta is not None:
            self.eta = eta
        if eta_test is not None:
            self.eta_test = eta_test


class SymmetricPositiveDefinite(Manifold):
    """
    Subclass of the SymmetricPositiveDefinite manifold using the
    affine invariant Riemannian metric (AIRM) as default metric
    """

    __scaling__ = Manifold.__scaling__.copy()
    name = "SymmetricPositiveDefinite"
    ndim = 2
    reversible = False

    def __init__(self):
        super().__init__()

    def dist(self, x: torch.Tensor, y: torch.Tensor, keepdim) -> torch.Tensor:
        """
        Computes the affine invariant Riemannian metric (AIM)
        """
        inv_sqrt_x = functionals.sym_invsqrtm.apply(x)
        return torch.norm(
            functionals.sym_logm.apply(inv_sqrt_x @ y @ inv_sqrt_x),
            dim=[-1, -2],
            keepdim=keepdim,
        )

    def _check_point_on_manifold(
            self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        ok = torch.allclose(x, x.transpose(-1, -2), atol=atol, rtol=rtol)
        if not ok:
            return False, "`x != x.transpose` with atol={}, rtol={}".format(atol, rtol)
        e = torch.linalg.eigvalsh(x)
        ok = (e > -atol).min()
        if not ok:
            return False, "eigenvalues of x are not all greater than 0."
        return True, None

    def _check_vector_on_tangent(
            self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        ok = torch.allclose(u, u.transpose(-1, -2), atol=atol, rtol=rtol)
        if not ok:
            return False, "`u != u.transpose` with atol={}, rtol={}".format(atol, rtol)
        return True, None

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        symx = functionals.ensure_sym(x)
        return functionals.sym_abseig.apply(symx)

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return functionals.ensure_sym(u)

    def egrad2rgrad(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return x @ self.proju(x, u) @ x

    def inner(self, x: torch.Tensor, u: torch.Tensor, v: Optional[torch.Tensor], keepdim) -> torch.Tensor:
        if v is None:
            v = u
        inv_x = functionals.sym_invm.apply(x)
        ret = torch.diagonal(inv_x @ u @ inv_x @ v, dim1=-2, dim2=-1).sum(-1)
        if keepdim:
            return torch.unsqueeze(torch.unsqueeze(ret, -1), -1)
        return ret

    def retr(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        inv_x = functionals.sym_invm.apply(x)
        return functionals.ensure_sym(x + u + 0.5 * u @ inv_x @ u)
        # return self.expmap(x, u)

    def expmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        sqrt_x, inv_sqrt_x = functionals.sym_invsqrtm2.apply(x)
        return sqrt_x @ functionals.sym_expm.apply(inv_sqrt_x @ u @ inv_sqrt_x) @ sqrt_x

    def logmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        sqrt_x, inv_sqrt_x = functionals.sym_invsqrtm2.apply(x)
        return sqrt_x @ functionals.sym_logm.apply(inv_sqrt_x @ u @ inv_sqrt_x) @ sqrt_x

    def extra_repr(self) -> str:
        return "default_metric=AIM"

    def transp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:

        xinvy = torch.linalg.solve(x.double(), y.double())
        s, U = torch.linalg.eig(xinvy.transpose(-2, -1))
        s = s.real
        U = U.real

        Ut = U.transpose(-2, -1)
        Esqm = torch.linalg.solve(Ut, torch.diag_embed(s.sqrt()) @ Ut).transpose(-2, -1).to(y.dtype)

        return Esqm @ v @ Esqm.transpose(-1, -2)

    def random(self, *size, dtype=None, device=None, **kwargs) -> torch.Tensor:
        tens = torch.randn(*size, dtype=dtype, device=device, **kwargs)
        tens = functionals.ensure_sym(tens)
        tens = functionals.sym_expm.apply(tens)
        return tens

    def barycenter(self, X: torch.Tensor, steps: int = 1, dim=0) -> torch.Tensor:
        """
        Compute several steps of the Kracher flow algorithm to estimate the
        Barycenter on the manifold.
        """
        return functionals.spd_mean_kracher_flow(X, None, maxiter=steps, dim=dim, return_dist=False)

    def geodesic(self, A: torch.Tensor, B: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute geodesic between two SPD tensors A and B and return
        point on the geodesic at length t \in [0,1]
        if t = 0, then A is returned
        if t = 1, then B is returned
        """
        Asq, Ainvsq = functionals.sym_invsqrtm2.apply(A)
        return Asq @ functionals.sym_powm.apply(Ainvsq @ B @ Ainvsq, t) @ Asq

    def transp_via_identity(self, X: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Parallel transport of the tensors in X around A to the identity matrix I
        Parallel transport from around the identity matrix to the new center (tensor B)
        """
        Ainvsq = functionals.sym_invsqrtm.apply(A)
        Bsq = functionals.sym_sqrtm.apply(B)
        return Bsq @ (Ainvsq @ X @ Ainvsq) @ Bsq

    def transp_identity_rescale_transp(self, X: torch.Tensor, A: torch.Tensor, s: torch.Tensor,
                                       B: torch.Tensor) -> torch.Tensor:
        """
        Parallel transport of the tensors in X around A to the identity matrix I
        Rescales the dispersion by the factor s
        Parallel transport from the identity to the new center (tensor B)
        """
        Ainvsq = functionals.sym_invsqrtm.apply(A)
        Bsq = functionals.sym_sqrtm.apply(B)
        return Bsq @ functionals.sym_powm.apply(Ainvsq @ X @ Ainvsq, s) @ Bsq

    def transp_identity_rescale_rotate_transp(self, X: torch.Tensor, A: torch.Tensor, s: torch.Tensor, B: torch.Tensor,
                                              W: torch.Tensor) -> torch.Tensor:
        """
        Parallel transport of the tensors in X around A to the identity matrix I
        Rescales the dispersion by the factor s
        Parallel transport from the identity to the new center (tensor B)
        """
        Ainvsq = functionals.sym_invsqrtm.apply(A)
        Bsq = functionals.sym_sqrtm.apply(B)
        WBsq = W @ Bsq
        return WBsq.transpose(-2, -1) @ functionals.sym_powm.apply(Ainvsq @ X @ Ainvsq, s) @ WBsq


class SPDBatchNormImpl(BaseBatchNorm):
    def __init__(self, shape: Tuple[int, ...] or torch.Size, batchdim: int,
                 eta=1., eta_test=0.1,
                 karcher_steps: int = 1, learn_mean=True, learn_std=True,
                 dispersion: BatchNormDispersion = BatchNormDispersion.SCALAR,
                 eps=1e-5, mean=None, std=None, **kwargs):
        super().__init__(eta, eta_test)
        # the last two dimensions are used for SPD manifold
        assert (shape[-1] == shape[-2])

        if dispersion == BatchNormDispersion.VECTOR:
            raise NotImplementedError()

        self.dispersion = dispersion
        self.learn_mean = learn_mean
        self.learn_std = learn_std
        self.batchdim = batchdim
        self.karcher_steps = karcher_steps
        self.eps = eps

        init_mean = torch.diag_embed(torch.ones(shape[:-1], **kwargs))
        init_var = torch.ones((*shape[:-2], 1), **kwargs)

        self.register_buffer('running_mean', ManifoldTensor(init_mean,
                                                            manifold=SymmetricPositiveDefinite()))
        self.register_buffer('running_var', init_var)
        self.register_buffer('running_mean_test', ManifoldTensor(init_mean,
                                                                 manifold=SymmetricPositiveDefinite()))
        self.register_buffer('running_var_test', init_var)

        if mean is not None:
            self.mean = mean
        else:
            if self.learn_mean:
                self.mean = ManifoldParameter(init_mean.clone(), manifold=SymmetricPositiveDefinite())
            else:
                self.mean = ManifoldTensor(init_mean.clone(), manifold=SymmetricPositiveDefinite())

        if self.dispersion is not BatchNormDispersion.NONE:
            if std is not None:
                self.std = std
            else:
                if self.learn_std:
                    self.std = nn.parameter.Parameter(init_var.clone())
                else:
                    self.std = init_var.clone()

    @torch.no_grad()
    def initrunningstats(self, X):
        self.running_mean.data, geom_dist = functionals.spd_mean_kracher_flow(X, dim=self.batchdim, return_dist=True)
        self.running_mean_test.data = self.running_mean.data.clone()

        if self.dispersion is BatchNormDispersion.SCALAR:
            self.running_var = \
            geom_dist.square().mean(dim=self.batchdim, keepdim=True).clamp(min=functionals.EPS[X.dtype])[..., None]
            self.running_var_test = self.running_var.clone()

    def forward(self, X):
        manifold = self.running_mean.manifold
        if self.training:
            # compute the Karcher flow for the current batch
            batch_mean = X.mean(dim=self.batchdim, keepdim=True)
            for _ in range(self.karcher_steps):
                bm_sq, bm_invsq = functionals.sym_invsqrtm2.apply(batch_mean.detach())
                XT = functionals.sym_logm.apply(bm_invsq @ X @ bm_invsq)
                GT = XT.mean(dim=self.batchdim, keepdim=True)
                batch_mean = bm_sq @ functionals.sym_expm.apply(GT) @ bm_sq

            # update the running mean
            rm = functionals.spd_2point_interpolation(self.running_mean, batch_mean, self.eta)

            if self.dispersion is BatchNormDispersion.SCALAR:
                GT = functionals.sym_logm.apply(bm_invsq @ rm @ bm_invsq)
                batch_var = torch.norm(XT - GT, p='fro', dim=(-2, -1), keepdim=True).square().mean(dim=self.batchdim,
                                                                                                   keepdim=True).squeeze(
                    -1)
                rv = (1. - self.eta) * self.running_var + self.eta * batch_var

        else:
            if self.test_stats_mode == BatchNormTestStatsMode.BUFFER:
                pass  # nothing to do: use the ones in the buffer
            elif self.test_stats_mode == BatchNormTestStatsMode.REFIT:
                self.initrunningstats(X)
            elif self.test_stats_mode == BatchNormTestStatsMode.ADAPT:
                raise NotImplementedError()

            rm = self.running_mean_test
            if self.dispersion is BatchNormDispersion.SCALAR:
                rv = self.running_var_test

        # rescale to desired dispersion
        if self.dispersion is BatchNormDispersion.SCALAR:
            Xn = manifold.transp_identity_rescale_transp(X,
                                                         rm, self.std / (rv + self.eps).sqrt(), self.mean)
        else:
            Xn = manifold.transp_via_identity(X, rm, self.mean)

        if self.training:
            with torch.no_grad():
                self.running_mean.data = rm.clone()
                self.running_mean_test.data = functionals.spd_2point_interpolation(self.running_mean_test, batch_mean,
                                                                                   self.eta_test)
                if self.dispersion is not BatchNormDispersion.NONE:
                    self.running_var = rv.clone()
                    GT_test = functionals.sym_logm.apply(bm_invsq @ self.running_mean_test @ bm_invsq)
                    batch_var_test = torch.norm(XT - GT_test, p='fro', dim=(-2, -1), keepdim=True).square().mean(
                        dim=self.batchdim, keepdim=True).squeeze(-1)

                    self.running_var_test = (
                                                        1. - self.eta_test) * self.running_var_test + self.eta_test * batch_var_test

        return Xn


class AdaMomSPDBatchNorm(SPDBatchNormImpl,SchedulableBatchNorm):
    """
    Adaptive momentum batch normalization on the SPD manifold [proposed].

    The momentum terms can be controlled via a momentum scheduler.
    """
    def __init__(self, shape: Tuple[int, ...] or torch.Size, 
                 batchdim: int,
                 eta=1.0, eta_test=0.1, **kwargs):
        super().__init__(shape=shape, batchdim=batchdim, 
                         eta=eta, eta_test=eta_test, **kwargs)