from .base import SPDNet


class FBSPDNet(SPDNet):
    # just here for compatability purposes
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        x = super().forward(x)
        return x
