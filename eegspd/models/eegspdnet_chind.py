from .base import SPDNet
from ..modules import ChIndConv


class EEGSPDNet_ChInd(SPDNet):
    def __init__(self, filter_time_length=25, **kwargs):
        super().__init__(**kwargs)
        self.filter_time_length = filter_time_length
        self.conv = ChIndConv(n_filters=self.n_filters, filter_time_length=filter_time_length)

    def forward(self, x):
        x = self.conv(x)
        x = super().forward(x)
        return x
