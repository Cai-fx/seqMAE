from seqMAE import preprocessing as pp
from seqMAE import utils as utils
from seqMAE import core as core
from seqMAE import data as data 


import sys
sys.modules.update({f"{__name__}.{m}": globals()[m] for m in ["pp", "utils", "core", "data"]})
