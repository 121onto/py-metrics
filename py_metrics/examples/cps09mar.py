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
    frame['intercept'] = 1.0

    frame['married'] = frame['marital'].isin([1,2,3]).astype(int)
    frame['formerly_married'] = frame['marital'].isin([4,5,6]).astype(int)

    frame['female*union'] = frame['female'] * frame['union']
    frame['male*union'] = (1 - frame['female']) * frame['union']
    frame['female*married'] = frame['female'] * frame['married']
    frame['male*married'] = (1 - frame['female']) * frame['married']
    frame['female*fromerly_married'] = frame['female'] * frame['formerly_married']
    frame['male*formerly_married'] = (1 - frame['female']) * frame['formerly_married']

    frame['hispanic'] = frame['hisp']
    frame['black'] = (frame['race'] == 2).astype(int)
    frame['american_indian'] = (frame['race'] == 3).astype(int)
    frame['asian'] = (frame['race'] == 4).astype(int)
    frame['mixed_race'] = (frame['race'] >= 6).astype(int)

    frame['wage'] = frame['earnings'] / (frame['week'] * frame['hours'])
    frame['log(wage)'] = np.log(frame['wage'])

    frame['experience'] = np.maximum(frame['age'] - frame['education'] - 6, 0)
    frame['experience^2'] = frame['experience'] ** 2
    frame['experience^2/100'] = frame['experience'] ** 2 / 100
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


def filter_3(frame):
    # Chapter 3.7 of Hansen's text (January 2019 edition)
    frame = frame.copy(deep=True)

    # Education
    mask = (frame['education'] >= 12)
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

    # SOURCE: chapter 4.15, page 122
    print(reg.std_err(estimator='v_0')) # OUT: [0.045, 0.707]
    print(reg.std_err(estimator='v_hc0')) # OUT: [0.029, 0.461]
    print(reg.std_err(estimator='v_hc1')) # OUT: [0.030, 0.486]
    print(reg.std_err(estimator='v_hc2')) # OUT: [0.031, 0.493]
    print(reg.std_err(estimator='v_hc3')) # OUT: [0.033, 0.527]


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


def reg_3():
    # SOURCE: Hansen, Chapters 4.19, page 126.
    filename = caches.data_path('cps09mar.txt')
    frame = pd.read_csv(filename)
    frame = featurize(frame)
    frame = filter_3(frame)

    y = 'log(wage)'
    x = [
        'education',
        'experience', 'experience^2/100',
        'female',
        'female*union',
        'male*union',
        'female*married',
        'male*married',
        'female*fromerly_married',
        'male*formerly_married',
        'hispanic',
        'black',
        'american_indian',
        'asian',
        'mixed_race',
        'intercept',
    ]

    reg = base.Reg(x, y)
    reg.fit(frame)

    # Chapter 4.19, pages 126-127
    reg.summarize()  # OUT: table 4.2, page 127
