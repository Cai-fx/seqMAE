from scb_multiome import preprocessing as pp
from scb_multiome import utils as utils
from scb_multiome import core as core
from scb_multiome import data as data 


import sys
sys.modules.update({f"{__name__}.{m}": globals()[m] for m in ["pp", "utils", "core", "data"]})
