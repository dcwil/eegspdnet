from .base import SPDNet
from ..modules import ChIndSincConv


class EEGSPDNet_ChInd_Sinc_RmInt(SPDNet):

    def __init__(
            self,
            sfreq,  # This needs to be the sf of the dataset
            filter_time_length=25,
            **kwargs,
    ):
        super().__init__(remove_interband=True, keep_inter_elec=False, **kwargs)
        self.sfreq = sfreq
        self.filter_time_length = filter_time_length

        # layers
        self.conv = ChIndSincConv(
            n_filters=self.n_filters,
            fs=self.sfreq,
            filter_length=self.filter_time_length,
            spacing='rand'
        )

    def forward(self, x):
        x = self.conv(x)
        x = super().forward(x)
        return x
