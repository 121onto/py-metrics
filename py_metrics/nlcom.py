"""Non-linear combinations of estimators.

Use these methods to compute variance-covariance matrices or standard errors
for non-linear combinations of estimators r(beta).  The methods in this
module take an argument `gradient` which is a function that computes the
gradient of `r(beta)` (a function of beta).  This module can also be used to
construct forecast intervals in a linear regression (although Hansen warns
against doing this).  See Hansen's chapter 7.15 for more details.
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import pandas as pd

###########################################################################
# VCE
###########################################################################

def vce(reg, gradient, estimator=None):
    """Asymptotic covariance matrix estimation.

    Parameters
    ----------
    reg: one of regress.Reg, regress.Cluster, or regress.CnsReg
        You must run `fit` on this object before passing it in.
    gradient: function or None
        The gradient of your functional, takes beta as an argument and
        returns an np.array of shpae (k,m).
    estimator: string or None
        Estimation method used to compute the vce in Reg or Cluster.  If the
        user does not provide a string here we use the default option.  See
        those classes for more details (optional).

    Returns
    -------
    float
    """
    r = gradient(reg.beta)
    vce = reg.vce() if estimator is None else reg.vce(estimator=estimator)
    return np.matmul(np.transpose(r), np.matmul(vce, r))


###########################################################################
# Standard errors
###########################################################################

def ve(reg, gradient, estimator=None):
    v = vce(reg, gradient, estimator=estimator)
    return v if v.ndim == 0 else np.diag(v)


def std_err(reg, gradient, estimator=None):
    return np.sqrt(ve(reg, gradient, estimator=estimator))


###########################################################################
# Forecast interval
###########################################################################

def forecast_std_err(reg, x, estimator=None):
    """Compute a forecast error for the fitted regression reg.

    Parameters
    ----------
    reg: one of regress.Reg, regress.Cluster, or regress.CnsReg
        You must run `fit` on this object before passing it in.
    x: np.array of type np.float32
        The value at which you would like to compute a forecast interval.
    estimator: string or None
        Estimation method used to compute the vce in Reg or Cluster.  If the
        user does not provide a string here we use the default option.  See
        those classes for more details (optional).

    Discussion
    ----------
    Use this to compute a forecast interval at x for your regression equal
    to x' beta +/- 2 * x' V x.

    WARNING: Hansen warns against using this calculation as there is no rigorous
    justification for doing so.
    """
    gradent = lambda beta: x
    return np.sqrt(reg.o_hat ** 2 + vce(reg, gradient, estimator=None))
