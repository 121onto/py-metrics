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
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.h = None

        self._is_fit = False
        self.beta = None
        self.sse = None
        self.rnk_beta = None
        self.svs_beta = None

        self.qxx = None
        self.qxy = None


    def fit(self, frame):
        # Prepare the data
        x = frame[self.x].copy(deep=True).astype(np.float32).values
        y = frame[self.y].copy(deep=True).astype(np.float32).valuse

        # other operations
        self.qxx = np.matmul(np.transpose(x), x)
        self.qxy = np.matmul(np.transpose(x), y)
        self.qxx_inv = inv(self.qxx)

        # Fit the regression
        self.beta, self.sse, self.rnk_beta, self.svs_beta = lstsq(x, y)
        self._is_fit = True

        # Compute summary stats
        self.ssy = ((y - y.mean()) ** 2).sum()
        self.omega_hat = self.see / frame.shape[0]
        self.r2 = (1 - self.sse / self.ssy)


    def predict(frame):
        x = frame[self.x].copy(deep=True).astype(np.float32).values
        return np.matmul(x, self.beta)


    def leverage(frame):
        # SOURCE: https://stackoverflow.com/a/39534036/759442
        if self.h is not None:
            return self.h
        x = frame[self.x].copy(deep=True).astype(np.float32).values
        qxx_u = cholesky(self.qxx)
        z = lstsq(np.transpose(qxx_u), np.transpose(x))
        self.h = np.square(z).sum(axis=1)
        return self.h


###########################################################################
# Covariance matrix estimation
###########################################################################

class AVar(object):
    def __init__(self):
        pass
