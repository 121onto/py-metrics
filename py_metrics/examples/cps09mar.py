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
    # Chapter 3.7 of Hansen's text (January 2019 edition)
    frame = frame.copy(deep=True)
    frame['intercept'] = 1.0

    frame['wage'] = frame['earnings'] / (frame['week'] * frame['hours'])
    frame['log(wage)'] = np.log(frame['wage'])

    frame['experience'] = np.maximum(frame['age'] - frame['education'] - 6, 0)
    frame['experience^2'] = frame['experience'] ** 2
    frame['education*log(wage)'] = frame['education'] * frame['log(wage)']

    return frame


###########################################################################
# Filter
###########################################################################

def filter_1(frame):
    # Chapter 3.7 of Hansen's text (January 2019 edition)
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


def filter_2(frame):
    # Chapter 3.7 of Hansen's text (January 2019 edition)
    frame = frame.copy(deep=True)
    # Single
    mask = (frame['marital'] == 7)
    frame = frame[mask]

    # Race - Asian
    mask = (frame['race'] == 4)
    frame = frame[mask]

    # Male
    mask = (frame['female'] == 0)
    frame = frame[mask]

    return frame


###########################################################################
# Regression
###########################################################################

def reg_1():
    # SOURCE: Hansen, Chapters 3.7, equation 3.13, page 75
    filename = caches.data_path('cps09mar.txt')
    frame = pd.read_csv(filename)
    frame = featurize(frame)
    frame = filter_1(frame)

    y = 'log(wage)'
    x = ['education', 'intercept']
    reg = base.Reg(x, y)
    reg.fit(frame)

    # SOURCE: chapter 4.15, page 121
    print(reg.vce(estimator='v_0')) # OUT: [[0.002, -0.031], [-0.031, 0.499]]
    print(reg.vce(estimator='v_hc2')) # OUT: [[ 0.0009314  -0.01479734], [-0.01479722  0.24282344]]


def reg_2():
    # SOURCE: Hansen, Chapters 3.7, equation 3.14, pages 76.
    filename = caches.data_path('cps09mar.txt')
    frame = pd.read_csv(filename)
    frame = featurize(frame)
    frame = filter_2(frame)

    y = 'log(wage)'
    x = ['intercept', 'education', 'experience', 'experience^2']
    reg = base.Reg(x, y)
    reg.fit(frame)

    # Chapter 3.21, page 92
    print(frame.shape) # OUT: (268, 18)
    print(reg.influence()) # OUT: 29%
