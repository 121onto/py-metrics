"""Methods for saving and loading objects."""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import functools
import numpy as np
import pandas as pd

from py_metrics import caches, base

###########################################################################
# Featurize
###########################################################################

def featurize(frame):
    frame = frame.copy(deep=True)
    frame['intercept'] = 1.0

    # extract normalized testscore
    std = frame['totalscore'].std()
    mu = frame['totalscore'].mean()
    frame['testscore']  = (frame['totalscore'] - mu) / std
    return frame


###########################################################################
# Filter
###########################################################################

def filter_1(frame):
    # Chapter 4.21 of Hansen's text (January 2019 edition)
    frame = frame.copy(deep=True)
    return frame


###########################################################################
# Regression
###########################################################################

def reg_1():
    # SOURCE: Hansen, Chapters 4.21, equation 3.4.41, page 130
    filename = caches.data_path('ddk2011.txt')
    frame = pd.read_csv(filename)
    frame = featurize(frame)
    frame = filter_1(frame)

    x = ['intercept', 'tracking']
    y = 'testscore'

    reg = base.Reg(x, y)
    reg.fit(frame)
    reg.summarize()


def reg_2():
    # SOURCE: Hansen, Chapters 4.21, equation 3.4.41, page 130
    filename = caches.data_path('ddk2011.txt')
    frame = pd.read_csv(filename)
    frame = featurize(frame)
    frame = filter_1(frame)

    x = ['intercept', 'tracking']
    y = 'testscore'
    grp = 'schoolid'

    reg = base.Cluster(x, y, grp)
    reg.fit(frame)
    reg.summarize()


###########################################################################
# Main
###########################################################################

if __name__ == '__main__':
    reg_1()
    reg_2()
