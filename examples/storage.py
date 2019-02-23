"""Methods for finding data."""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os

###########################################################################
# Directories
###########################################################################

this_dir = os.path.dirname(__file__)
data_dir = os.path.join(this_dir, '..', 'data')

###########################################################################
# Paths
###########################################################################

def data_path(filename):
    return os.path.join(data_dir, filename)
