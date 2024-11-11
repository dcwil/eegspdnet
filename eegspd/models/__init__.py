from braindecode.models import Deep4Net, ShallowFBCSPNet, EEGNetv4

from .eegspdnet_chspec import EEGSPDNet_ChSpec
from .eegspdnet_chind import EEGSPDNet_ChInd
from .eegspdnet_chind_rmint import EEGSPDNet_ChInd_RmInt
from .eegspdnet_chind_rmint_keepintel import EEGSPDNet_ChInd_RmInt_KeepIntEl
from .eegspdnet_chspec_sinc import EEGSPDNet_ChSpec_Sinc
from .eegspdnet_chind_sinc import EEGSPDNet_ChInd_Sinc
from .eegspdnet_chind_sinc_rmint import EEGSPDNet_ChInd_Sinc_RmInt
from .eegspdnet_chind_sinc_rmint_keepintel import EEGSPDNet_ChInd_Sinc_RmInt_KeepIntEl
from .fbspdnet import FBSPDNet
from .tsmnet import TSMNet
from .util import _init_models_dict

_init_models_dict()