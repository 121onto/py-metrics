"""Methods for saving and loading objects."""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import pkg_resources
import appdirs

import py_metrics

###########################################################################
# Directories
###########################################################################

cache_dir = appdirs.user_cache_dir(
    appname=py_metrics.__appname__,
    version=py_metrics.__version__)


data_dir = pkg_resources.resource_filename(
    py_metrics.__appname__,
    'data')


###########################################################################
# Paths
###########################################################################

def cache_path(filename):
    return os.path.join(cache_dir, filename)


def data_path(filename):
    return os.path.join(data_dir, filename)
