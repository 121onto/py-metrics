"""Regression in various forms."""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import pandas as pd

from numpy.linalg import inv as np_inv
from scipy.linalg import lstsq

###########################################################################
# OLS
###########################################################################

# placeholder pp 279

def _compute_ols(_sentinel=None, x=None, y=None, coefficient_only=False):
    """Compute hat matrices for ols regression.

    Parameters
    ----------
    _sentinel: (restricted)
        Do not use this parameter, it is here to enforce the use of named
        arguments.
    x: np.array of type np.float32 with shape (n, k)
        Observations on the explanatory variables.
    y: np.array of type np.float with shape (n,)
        Observations on the dependent variable.
    coefficient_only: boolean
        Whether or not we should to compute the coefficient only (default False).
        The function will be more efficient when this is set to True.

    Returns
    -------
    beta: np.array of type np.float32 and shape (k,)
        The ols estimate of the coefficient in a linear regression of x on y.
    sse: np.float32
        The sum of squared errors.
    ssy: np.float32
        The sum of squared deviations of y from it's mean.
    qxx: np.array of type np.float32 and shape (k,k)
        The hat-matrix Qxx, which is the empirical analogue of E[xx']
        where x is a vector of shape (k,) representing a single observation.
    qxy: np.array of type np.float32 and shape (k,)
        The hat-matrix Qxy, which is the empirical analogue of E[xy]
        where x is a vector of shape (k,) representing a single observation
        and y is a scalar representing a single observation on the dependent
        variable y.

    """
    if _sentinel is not None:
        raise ValueError('''
        `_compute_ols` accepts named kwargs only.''')
    if x is None or y is None:
        raise ValueError('''
        Both `x` and `y` must be defined in call to `_compute_ols`.''')

    # Fit the regression
    beta, sse, _, _ = lstsq(x, y)
    if coefficient_only:
        return beta, None, None, None, None, None

    ssy = ((y - y.mean()) ** 2).sum()

    qxx = np.einsum('ij,ik', x, x)
    qxx_inv = np_inv(qxx)
    qxy = np.einsum('ij,i', x, y)

    return beta, sse, ssy, qxx, qxx_inv, qxy
