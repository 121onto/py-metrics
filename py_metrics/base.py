"""Base classes."""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import pandas as pd

from scipy.linalg import lstsq
from numpy.linalg import inv
from numpy.linalg import cholesky

from py_metrics import caches

###########################################################################
# Estimator
###########################################################################

class Reg(object):
    def __init__(self, x_cols, y_col):
        # TODO (121onto): enforce types

        # Columns defining estimation
        self.x_cols = x_cols
        self.y_col = y_col

        # Data arraays
        self.x = None
        self.y = None
        self.h = None

        # Hat matrices
        self.qxx = None
        self.qxy = None
        self.qxx_inv = None

        # Coefficient estimates
        self.beta = None
        self._is_fit = False

        # Diagnostics
        self.e_hat = None
        self.e_til = None
        self.sse = None
        self.ssy = None
        self.omega_hat = None
        self.omega_til = None
        self.r2 = None


    def fit(self, frame):
        # Prepare the data
        self.x = x = frame[self.x_cols].copy(deep=True).astype(np.float32).values
        self.y = y = frame[self.y_col].copy(deep=True).astype(np.float32).values

        # other operations
        self.qxx = np.matmul(np.transpose(x), x)
        self.qxy = np.matmul(np.transpose(x), y)
        self.qxx_inv = inv(self.qxx)

        # Fit the regression
        self.beta, self.sse, _, _ = lstsq(x, y)
        self._is_fit = True

        # Errors
        self.e_hat = y - np.matmul(x, self.beta)
        self.e_til = ((1 - self.leverage()) ** -1 ) * self.e_hat

        # Summary stats
        self.ssy = ((y - y.mean()) ** 2).sum()
        self.omega_hat = self.sse / frame.shape[0]
        self.omega_til = ((self.e_til) ** 2).sum() / frame.shape[0]
        self.r2 = (1 - self.sse / self.ssy)


    def predict(self, frame):
        if not self._is_fit:
            raise RuntimeError('''
            You must run `Reg.fit` before running `Reg.predict`.''')

        x = frame[self.x].copy(deep=True).astype(np.float32).values
        return np.matmul(x, self.beta)


    def leverage(self):
        # SOURCE: https://stackoverflow.com/a/39534036/759442
        if not self._is_fit:
            raise RuntimeError('''
            You must run `Reg.fit` before running `Reg.leverage`.''')

        if self.h is not None:
            return self.h
        x = self.x
        qxx_u = cholesky(self.qxx)
        z, _, _, _ = lstsq(qxx_u, np.transpose(x))
        self.h = np.square(z).sum(axis=0)
        return self.h


    def summarize(self):
        if not self._is_fit:
            raise RuntimeError('''
            You must run `Reg.fit` before running `Reg.summarize`.''')

        x = self.x
        y = self.y

        # TODO (121onto): output a summary table


###########################################################################
# Covariance matrix estimation
###########################################################################

class AVar(object):
    def __init__(self):
        pass
