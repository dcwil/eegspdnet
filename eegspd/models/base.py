import torch.nn as nn

from .util import _parse_bimap_sizes, create_spdnet
from ..modules import SCMPool, RemoveInterbandCovariance
from ..functional import sym2vec_n_unique


class SPDNet(nn.Module):

    def __init__(
            self,
            n_chans,
            n_filters,
            n_classes,
            bimap_sizes=(2, 3),
            final_layer_drop_prob=0,
            spd_drop_prob=0,
            spd_drop_scaling=True,
            logeig_sqrt=True,
            remove_interband=False,
            keep_inter_elec=False,
            **kwargs,
    ):
        super().__init__()
        self.n_filters = n_filters
        self.n_elecs = n_chans  # braindecode refers to them as channels
        self.n_classes = n_classes
        self.bimap_sizes = bimap_sizes
        self.final_layer_drop_prob = final_layer_drop_prob
        self.logeig_sqrt = logeig_sqrt
        self.spd_drop_prob = spd_drop_prob
        self.spd_drop_scaling = spd_drop_scaling
        self.remove_interband = remove_interband
        self.keep_inter_elec = keep_inter_elec
        print(f'Model ignoring these passed args: {kwargs}')

        bimap_sizes_ls = _parse_bimap_sizes(bimap_sizes=bimap_sizes, n_filters=n_filters, n_elecs=self.n_elecs)

        # layers
        self.cov_pool = SCMPool()

        if self.remove_interband:
            print(f'Adding RmInt layer with {keep_inter_elec=}')
            self.rmint = RemoveInterbandCovariance(
                elecs_grouped=True,
                n_elecs=self.n_elecs,
                n_filters=self.n_filters,
                keep_inter_elec=self.keep_inter_elec
            )

        self.spdnet = create_spdnet(
            bimap_sizes=self.bimap_sizes,
            n_classes=self.n_classes,
            n_elecs=self.n_elecs,
            n_filters=self.n_filters,
            sqrt=self.logeig_sqrt,
            dropout=self.spd_drop_prob,
            dropout_scaling=self.spd_drop_scaling
        )
        self.dropout = nn.Dropout(p=self.final_layer_drop_prob)
        self.linear = nn.Linear(sym2vec_n_unique(ndim=bimap_sizes_ls[-1]), n_classes)

    def forward(self, x):
        x = self.cov_pool(x)
        if self.remove_interband:
            x = self.rmint(x)
        x = self.spdnet(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x
