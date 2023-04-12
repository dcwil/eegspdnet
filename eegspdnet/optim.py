from spdnet import StiefelParameter
from spdnet.utils import retraction, orthogonal_projection
from torchspdnet.optimizers import proj_tanX_stiefel, gram_schmidt


class AltStiefelMetaOptimizer(
    object
):  # adapted for toggling the weird 0 set of the weights
    """This is a meta optimizer which uses other optimizers for updating parameters
    and remap all StiefelParameter parameters to Stiefel space after they have been updated.

    set_zero: True=Euclidean grad, False=Rie Grad
    reset_zero: True: retract with correct term, False: Retract with correct term plus old weight (?)

    use_alt_proj and use_gram_schmidt were for verifying DK method against I

    For proper implementation want set_zero=False, reset_zero=True
    """

    def __init__(
        self,
        optimizer,
        set_zero=False,
        reset_zero=True,
        use_gram_schmidt=False,
        use_alt_proj=False,
    ):
        self.optimizer = optimizer
        self.state = {}
        self.set_zero = set_zero
        self.reset_zero = reset_zero
        self.use_gram_schmidt = use_gram_schmidt
        self.use_alt_proj = use_alt_proj

    def zero_grad(self):
        return self.optimizer.zero_grad()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if isinstance(p, StiefelParameter):
                    if id(p) not in self.state:
                        self.state[id(p)] = p.data.clone()
                    else:
                        self.state[id(p)].fill_(0).add_(p.data)

                    if self.set_zero:
                        p.data.fill_(0)

                    if self.use_alt_proj:
                        trans = proj_tanX_stiefel(
                            p.grad.data.unsqueeze(0).unsqueeze(0),
                            p.data.unsqueeze(0).unsqueeze(0),
                        ).squeeze()
                    else:
                        trans = orthogonal_projection(p.grad.data, p.data)

                    p.grad.data.fill_(0).add_(trans)

                    if self.reset_zero:
                        p.data.fill_(0)

        loss = self.optimizer.step(closure)

        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if isinstance(p, StiefelParameter):
                    if self.use_gram_schmidt:
                        trans = gram_schmidt(p.data + self.state[id(p)])[0]
                    else:
                        trans = retraction(p.data, self.state[id(p)])

                    p.data.fill_(0).add_(trans)

        return loss
