"""Methods for saving and loading objects."""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import functools
import numpy as np
import pandas as pd

from py_metrics import caches
from py_metrics import base
from py_metrics import utils

###########################################################################
# Featurize
###########################################################################

def featurize(frame):
    frame = frame.copy(deep=True)
    frame['wage'] = frame['earnings'] / (frame['week'] * frame['hours'])
    frame['log(wage)'] = np.log(frame['wage'])
    frame['experience'] = np.maximum(frame['age'] - frame['education'] - 6, 0)

    frame['education^2'] = frame['education'] ** 2
    frame['education*log(wage)'] = frame['education'] * frame['log(wage)']
    frame['intercept'] = 1.0
    return frame


###########################################################################
# Filter
###########################################################################

def filter(frame):
    frame = frame.copy(deep=True)
    # Married - civilian spouse present
    mask = (
        (frame['marital'] == 1) |
        (frame['marital'] == 2))
    # mask = (frame['marital'] == 1)
    frame = frame[mask]

    # Race - black
    mask = (
        (frame['race'] == 2) |
        (frame['race'] == 10) |
        (frame['race'] == 11) |
        (frame['race'] == 12))
    # mask = (frame['race'] == 2)
    frame = frame[mask]

    # Female
    mask = (frame['female'] == 1)
    frame = frame[mask]

    # Wage earner
    mask = (frame['earnings'] > 0)
    frame = frame[mask]

    # Work experience
    mask = (frame['experience'] == 12)
    frame = frame[mask]

    return frame


###########################################################################
# Regression
###########################################################################

def reg():
    filename = caches.data_path('cps09mar.txt')
    frame = pd.read_csv(filename)
    frame = featurize(frame)
    frame = filter(frame)

    y = 'log(wage)'
    x = ['education', 'intercept']
    reg = base.Reg(x, y)
    reg.fit(frame)
