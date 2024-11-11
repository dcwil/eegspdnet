import torch
import torch.nn as nn

from ..batchnorm import AdaMomSPDBatchNorm, BatchNormTestStatsMode, BatchNormDispersion
from ..modules import CovariancePool, BiMap, ReEig, LogEig


class TSMNet(nn.Module):
    """Adapting from https://github.com/rkobler/TSMNet/blob/main/spdnets/models/tsmnet.py"""

    def __init__(
            self,
            n_chans,
            n_classes,
            n_times,  # for compatability
            n_filters=4,
            spatial_filters=40,
            subspacedims=20, # TODO: is this first bimap size
            filter_time_length=25,
            # bnorm='spdbn', #TODO: investigate
    ):

        super().__init__()

        self.nchannels_ = n_chans
        self.nclasses_ = n_classes

        self.temporal_filters_ = n_filters
        self.spatial_filters_ = spatial_filters
        self.subspacedims = subspacedims
        self.temp_cnn_kernel = filter_time_length
        # self.bnorm_ = bnorm
        self.spd_device_ = torch.device('cpu')

        # TODO: bnorm dispersion??

        # TODO: replace with our helper fn?
        tsdim = int(subspacedims*(subspacedims+1)/2)

        self.cnn = torch.nn.Sequential(
            nn.Conv2d(1, self.temporal_filters_, kernel_size=(1, filter_time_length),
                            padding='same', padding_mode='reflect'),
            nn.Conv2d(self.temporal_filters_, self.spatial_filters_, (self.nchannels_, 1)),
            nn.Flatten(start_dim=2),
        )

        self.cov_pooling = nn.Sequential(CovariancePool())

        # BN
        self.spdbnorm = AdaMomSPDBatchNorm((1, subspacedims, subspacedims), batchdim=0,
                                             dispersion=BatchNormDispersion.SCALAR,
                                             learn_mean=False, learn_std=True,
                                             eta=1., eta_test=.1, dtype=torch.double, device=self.spd_device_)

        self.spdnet = torch.nn.Sequential(
            BiMap((1, self.spatial_filters_, subspacedims), dtype=torch.double, device=self.spd_device_),
            ReEig(threshold=5e-4),  # Use our threshold or not?
        )
        self.logeig = torch.nn.Sequential(
            LogEig(subspacedims),
            torch.nn.Flatten(start_dim=1),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(tsdim, self.nclasses_).double(),
        )

    def forward(self, x, return_latent=False, return_prebn=False, return_postbn=False):
        out = ()
        h = self.cnn(x.to(device='cpu')[:,None,...])
        C = self.cov_pooling(h).to(device=self.spd_device_, dtype=torch.double)
        l = self.spdnet(C)
        out += (l,) if return_prebn else ()

        l = self.spdbnorm(l) if hasattr(self, 'spdbnorm') else l
        # l = self.spddsbnorm(l, d.to(device=self.spd_device_)) if hasattr(self, 'spddsbnorm') else l
        out += (l,) if return_postbn else ()
        l = self.logeig(l)
        # l = self.tsbnorm(l) if hasattr(self, 'tsbnorm') else l
        # l = self.tsdsbnorm(l,d) if hasattr(self, 'tsdsbnorm') else l

        out += (l,) if return_latent else ()
        y = self.classifier(l)
        out = y if len(out) == 0 else (y, *out[::-1])
        return out
    
    def finetune(self, x, y, d):
        if hasattr(self, 'spdbnorm'):
            self.spdbnorm.set_test_stats_mode(BatchNormTestStatsMode.REFIT)
        if hasattr(self, 'tsbnorm'):
            self.tsbnorm.set_test_stats_mode(BatchNormTestStatsMode.REFIT)

        with torch.no_grad():
            self.forward(x, d)

        if hasattr(self, 'spdbnorm'):
            self.spdbnorm.set_test_stats_mode(BatchNormTestStatsMode.BUFFER)
        if hasattr(self, 'tsbnorm'):
            self.tsbnorm.set_test_stats_mode(BatchNormTestStatsMode.BUFFER)

    def compute_patterns(self, x, y, d):
        pass