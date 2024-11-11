from .base import SPDNet
from ..modules import ChIndConv


class EEGSPDNet_ChInd_RmInt(SPDNet):
    def __init__(self, filter_time_length=25, **kwargs):
        super().__init__(remove_interband=True, keep_inter_elec=False, **kwargs)
        self.filter_time_length = filter_time_length
        self.conv = ChIndConv(n_filters=self.n_filters, filter_time_length=filter_time_length)

    def forward(self, x):
        x = self.conv(x)
        x = super().forward(x)
        return x
