"""Methods for saving and loading objects."""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import functools
import numpy as np
import pandas as pd

from py_metrics import caches, regress

###########################################################################
# Generate
###########################################################################

def generate():
    """Generate data."""
