import torch
from einops import einsum
from sklearn.base import BaseEstimator, TransformerMixin

from ..functional import create_interband_mask, sym_reeig


class RemoveInterbandCovariance_sk(BaseEstimator, TransformerMixin):
    def __init__(
        self, elecs_grouped, n_filters, n_elecs, keep_inter_elec=False
    ):
        self.elecs_grouped = elecs_grouped
        self.n_filters = n_filters
        self.n_elecs = n_elecs
        self.keep_inter_elec = keep_inter_elec
        self.mask = create_interband_mask(
            elecs_grouped=elecs_grouped,
            n_filters=n_filters,
            n_elecs=n_elecs,
            keep_inter_elec=keep_inter_elec,
        ).numpy()

    def fit(self, X, y=None):
        # do nothing
        return self

    def transform(self, X):
        X = einsum(X, self.mask, "b c1 c2, c1 c2 -> b c1 c2")
        return X


class ReEig_sk(BaseEstimator, TransformerMixin):
    # NOTE: edited the typical 1e-4 threshold due to some computation issues
    def __init__(self, epsilon=torch.Tensor([5e-4])):
        self.epsilon = epsilon

    def fit(self, X, y=None):
        # do nothing
        return self

    def transform(self, X):
        X = torch.from_numpy(X)
        X = sym_reeig.apply(X, self.epsilon)
        return X.numpy()
