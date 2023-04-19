import torch
from torch.autograd import Function

from spdnet.utils import symmetric


class LogEig_I(Function):
    """Copies https://github.com/adavoudi/spdnet/blob/2a15e908634cd8db6c75ea45d9e3bd567203eccf/spdnet/spd.py#L126-L175
    for Ionescu method.
    Adapts https://gitlab.lip6.fr/schwander/torchspdnet/-/blob/master/torchspdnet/functional.py for DK method
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)

        output = input.new(input.size(0), input.size(1), input.size(2))
        for k, x in enumerate(input):
            u, s, v = x.svd()
            s.log_()
            output[k] = u.mm(s.diag().mm(u.t()))

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_variables
        input = input[0]
        grad_input = None

        if ctx.needs_input_grad[0]:
            eye = input.new(input.size(1))
            eye.fill_(1)
            eye = eye.diag()
            grad_input = input.new(input.size(0), input.size(1), input.size(1))
            for k, dx in enumerate(grad_output):
                x = input[k]
                u, s, v = x.svd()

                dx = symmetric(dx)

                s_log_diag = s.log().diag()
                s_inv_diag = (1 / s).diag()

                dLdV = 2 * (dx.mm(u.mm(s_log_diag)))
                dLdS = eye * (s_inv_diag.mm(u.t().mm(dx.mm(u))))

                P = calculate_P(s, mode="I")

                grad_input[k] = u.mm(symmetric(P.t() * (u.t().mm(dLdV))) + dLdS).mm(
                    u.t()
                )

        return grad_input


class LogEig_DK(Function):
    """Copies https://github.com/adavoudi/spdnet/blob/2a15e908634cd8db6c75ea45d9e3bd567203eccf/spdnet/spd.py#L126-L175
    for Ionescu method.
    Adapts https://gitlab.lip6.fr/schwander/torchspdnet/-/blob/master/torchspdnet/functional.py for DK method
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)

        output = input.new(input.size(0), input.size(1), input.size(2))
        for k, x in enumerate(input):
            u, s, v = x.svd()
            s.log_()
            output[k] = u.mm(s.diag().mm(u.t()))

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_variables
        input = input[0]
        grad_input = None

        if ctx.needs_input_grad[0]:
            eye = input.new(input.size(1))
            eye.fill_(1)
            eye = eye.diag()
            grad_input = input.new(input.size(0), input.size(1), input.size(1))
            for k, dx in enumerate(grad_output):
                x = input[k]
                u, s, v = x.svd()

                P = calculate_P(s, mode="DK")

                dLdx = torch.einsum("ji, jk, kl, il, mi, nl -> mn", u, dx, u, P, u, u)
                grad_input[k] = dLdx

        return grad_input


class ReEig_I(Function):
    @staticmethod
    def forward(ctx, input, epsilon):
        ctx.save_for_backward(input, epsilon)

        output = input.new(input.size(0), input.size(1), input.size(2))
        for k, x in enumerate(input):
            u, s, v = x.svd()
            s[s < epsilon[0]] = epsilon[0]
            output[k] = u.mm(s.diag().mm(u.t()))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, epsilon = ctx.saved_variables
        grad_input = None

        if ctx.needs_input_grad[0]:
            eye = input.new(input.size(1))
            eye.fill_(1)
            eye = eye.diag()
            grad_input = input.new(input.size(0), input.size(1), input.size(2))
            for k, dx in enumerate(grad_output):
                if len(dx.shape) == 1:
                    continue

                dx = symmetric(dx)

                x = input[k]
                u, s, v = x.svd()

                max_mask = s > epsilon
                s_max_diag = s.clone()
                s_max_diag[~max_mask] = epsilon
                s_max_diag = s_max_diag.diag()
                Q = (
                    max_mask.float().diag()
                )  # for DK this is done in the calculate P function

                dLdV = 2 * (dx.mm(u.mm(s_max_diag)))
                dLdS = eye * (Q.mm(u.t().mm(dx.mm(u))))

                P = calculate_P(s, mode="I")

                grad_input[k] = u.mm(symmetric(P.t() * u.t().mm(dLdV)) + dLdS).mm(u.t())

        return grad_input, None


class ReEig_DK(Function):
    @staticmethod
    def forward(ctx, input, epsilon):
        ctx.save_for_backward(input, epsilon)

        output = input.new(input.size(0), input.size(1), input.size(2))
        for k, x in enumerate(input):
            u, s, v = x.svd()
            s[s < epsilon[0]] = epsilon[0]
            output[k] = u.mm(s.diag().mm(u.t()))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, epsilon = ctx.saved_variables
        grad_input = None

        if ctx.needs_input_grad[0]:
            eye = input.new(input.size(1))
            eye.fill_(1)
            eye = eye.diag()
            grad_input = input.new(input.size(0), input.size(1), input.size(2))
            for k, dx in enumerate(grad_output):
                if len(dx.shape) == 1:
                    continue

                x = input[k]
                u, s, v = x.svd()

                P = calculate_P(s, mode="DK", epsilon_DK=epsilon)

                dLdx = torch.einsum("ji, jk, kl, il, mi, nl -> mn", u, dx, u, P, u, u)
                grad_input[k] = dLdx

        return grad_input, None


def calculate_P(s, mode="I", epsilon_DK=None):
    """Called G in Engin et al. and called L in torchspdnet library"""

    s = s.squeeze()
    S = s.unsqueeze(1)
    S = S.expand(-1, S.size(0))

    if mode == "I":
        # Eq 14 Huang et al.
        P = S - torch.einsum("ij -> ji", S)
        mask_zero = torch.abs(P) == 0
        P = 1 / P
        P[mask_zero] = 0
    elif mode == "DK":
        if epsilon_DK is not None:  # provide epsilon for ReEig with DK case
            f_of_S = nn.Threshold(epsilon_DK[0], epsilon_DK[0])(S)
            df_of_S = S > epsilon_DK[0]
        else:
            f_of_S = S.log()
            df_of_S = 1 / S

        # Eq 12 Engin et al.
        P = (f_of_S - torch.einsum("ij -> ji", f_of_S)) / (
            S - torch.einsum("ij -> ji", S)
        )

        P[P == -np.inf] = 0
        P[P == np.inf] = 0
        P[torch.isnan(P)] = 0

        df_of_S = df_of_S * torch.eye(s.shape[0])  # turn into diag matrix

        P = P + df_of_S
    else:
        raise ValueError

    return P


def _regularise_with_oas_pytorch(matrix, n_samples, n_features):
    """Recreate regularise with oas func in pytorch"""
    trc = matrix.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)  # https://discuss.pytorch.org/t/get-the-trace-for-a-batch-of-matrices/108504
    mu = trc / n_features

    alpha = (matrix ** 2).view(matrix.shape[0], matrix.shape[1] * matrix.shape[1]).mean(axis=1)
    num = alpha + mu ** 2
    den = (n_samples + 1.) * (alpha - (mu ** 2) / n_features)
    shrinkage = torch.clip(num / den, max=1)
    shrunk_cov = (1. - shrinkage).view(matrix.shape[0], 1, 1) * matrix
    k = (shrinkage * mu).repeat_interleave(n_features).view(matrix.shape[0], n_features)
    shrunk_cov.diagonal(dim1=-2, dim2=-1)[:] += k  # https://discuss.pytorch.org/t/operation-on-diagonals-of-matrix-batch/50779

    return shrunk_cov