"""Regression in various forms."""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import pandas as pd

from numpy.linalg import inv as np_inv
from scipy.linalg import lstsq

###########################################################################
# Least squares
###########################################################################

# placeholder pp 279

def _least_squares(_sentinel=None, x=None, y=None):
    """Computes the least squares regression coefficient.

    Parameters
    ----------
    _sentinel: (restricted)
        Do not use this parameter, it is here to enforce the use of named
        arguments.
    x: np.array of type np.float32 with shape (n, k)
        Observations on the explanatory variables.
    y: np.array of type np.float with shape (n,)
        Observations on the dependent variable.


    Returns
    -------
    beta: np.array of type np.float32 and shape (k,)
        The ols estimate of the coefficient in a linear regression of x on y.
    sse: np.float32
        The sum of squared errors.
    """
    if _sentinel is not None:
        raise ValueError('''
        `_least_squares` accepts named kwargs only.''')
    if x is None or y is None:
        raise ValueError('''
        Both `x` and `y` must be defined in call to `_least_squares`.''')

    # Fit the regression
    beta, sse, _, _ = lstsq(x, y)
    return beta, sse


def _ssy(_sentinel=None, y=None):
    """Compute the sum of squared deviations of y from it's mean.

    Parameters
    ----------
    _sentinel: (restricted)
        Do not use this parameter, it is here to enforce the use of named
        arguments.
    y: np.array of type np.float with shape (n,)
        Observations on the dependent variable.


    Returns
    -------
    ssy: np.float32
        The sum of squared deviations of y from it's mean.

    """
    if _sentinel is not None:
        raise ValueError('''
        `_ssy` accepts named kwargs only.''')

    if y is None:
        raise ValueError('''
        `y` must be defined in call to `_ssy`.''')

    return ((y - y.mean()) ** 2).sum()


def _sandwich(_sentinel=None, x=None, w=None):
    """Computes the sandwich form x'wx.

    Parameters
    ----------
    _sentinel: (restricted)
        Do not use this parameter, it is here to enforce the use of named
        arguments.
    x: np.array of type np.float with shape (k,m)
        The matrix on the ouside of a sandwich form.
    w: np.array of type np.float with shape (k,k) or (k,) (optional)
        The matrix on the inside of a sandwich form.  Uses the identity
        matrix in case w is not passed.  Uses np.diag(w) in case
        a vector w is passed.

    Returns
    -------
    x'wx: np.array of type np.float32 and shape (m,m)
        The sandwich form built from x and w.
    """
    if _sentinel is not None:
        raise ValueError('''
        `_sandwich` accepts named kwargs only.''')

    if x is None:
        raise ValueError('''
        `x` must be defined in call to `_sandwich`.''')

    if w is None:
        sandwich = np.einsum('ij,ik', x, x)
    elif len(w.shape) == 1:
        sandwich = np.einsum(
            'ij,jk',
            np.multiply(np.transpose(x), w),
            np.multiply(w[:, np.newaxis], x)
        )
    else:
        sandwich = np.einsum('ij,jk', np.einsum('ij,ik', x, w), x)

    return sandwich
