"""HarDNet-MSEG (Apache-2.0) — minimal vendored model for colon polyp segmentation.

Upstream: https://github.com/james128333/HarDNet-MSEG
"""

from .hard_mseg import HarDMSEG
from .wrapper import HarDMSEGTwoClass

# Environment variable read by hardnet_68.hardnet() when no explicit path is passed.
HARDNET68_PRETRAINED_PATH_ENV = "HARDNET68_PRETRAINED_PATH"

__all__ = ["HarDMSEG", "HarDMSEGTwoClass", "HARDNET68_PRETRAINED_PATH_ENV"]
