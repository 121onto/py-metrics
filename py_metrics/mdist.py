"""Minimum distance estiamtion."""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import pandas as pd

from scipy.linalg import lstsq
from numpy.linalg import inv

from py_metrics import core

###########################################################################
# Minimum distnace
###########################################################################

class MinDist(object):
    """Minimum distance estimation.

    Discussion
    ----------
    A minimum distance estimator tries to find a parameter value which satisfies
    the constraint which is as close as possible to the unconstrained estimate.

    WARNING: underconstruction.
    """
    def __init__(self, x_cols, y_col, r=None, c=None):

        # Columns defining estimation
        self.x_cols = x_cols
        self.y_col = y_col
        self.k = len(x_cols)
        self.n = None
        self.q = None if c is None else c.shape[0]

        # Restrictions
        self.r = r
        self.c = c

        # Data arraays
        self.x = None
        self.y = None

        # Weight matrices
        self.w = None
        self.w_inv = None

        # Hat matrices
        self.qxx = None
        self.qxx_inv = None
        self.wi_r = None
        self.r_wi_r = None

        # Coefficient estimates
        self.beta_ols = None
        self.beta = None
        self._is_fit = False

        # Diagnostics
        # self.sse = None
        # self.ssy = None
        # ...

        # Error variance estimators
        # ...


    def residuals(self):
        if not self._is_fit:
            raise RuntimeError('''
            You must run `fit` before calling `residuals`.''')

        # Compute residuals
        x, y = self.x, self.y
        n, k, q = self.n, self.k, self.q

        self.ssy = core._ssy(y=y)
        # ...


    def fit(self, frame, r=None, c=None, weight_matrix='optimal'):
        """Regression for constrained least squares.

        Parameters
        ----------
        frame: pd.DataFrame with columns of type np.float32
            The data.
        r: np.array of type np.float32 of shape (k,q)
            The constraint matrix (optional if set during initialization).
        c: np.array of type np.float32 of shape (q,)
            The constraint value such that R'beta = c (optional if set during initialization).
        weight_matrix: string
            One of 'optimal', 'cls', ... (defaults to 'optimal').

        Discussion
        ----------
        """
        self.r = r = self.r if r is None else r
        self.c = self.c if c is None else c
        self.q = self.q if c is None else c.shape[0]

        self.x = x = frame[self.x_cols].copy(deep=True).astype(np.float32).values
        self.y = y = frame[self.y_col].copy(deep=True).astype(np.float32).values
        self.n = frame.shape[0]
        n, k, q = self.n, self.k, self.q

        # ols operations
        self.beta_ols, _ = core._least_squares(x=x, y=y)
        self.ssy = core._ssy(y=y)
        self.qxx = core._sandwich(x=x)
        self.qxx_inv = inv(qxx)

        if weight_matrix == 'cls':
            self.w = qxx
            self.w_inv = qxx_inv
        elif weight_matrix == 'optimal':
            pass
        else:
            raise NotImplementedError

        # cls operations
        self.wi_r = wi_r = np.matmul(self.w_inv, r)
        self.r_wi_r = r_wi_r = np.matmul(r, wi_r)
        rb = np.dot(np.transpose(r), self.beta_ols)
        rhs,_,_,_ = lstsq(r_wi_r, rb - c)
        lhs = wi_r

        # Fit the constrained regression
        self.beta = self.beta_ols - np.matmul(lhs, rhs)
        self._is_fit = True

        self.residuals()


    def predict(self, frame):
        if not self._is_fit:
            raise RuntimeError('''
            You must run `fit` before calling `predict`.''')

        x = frame[self.x_cols].copy(deep=True).astype(np.float32).values
        return np.matmul(x, self.beta)
