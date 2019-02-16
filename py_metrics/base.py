"""Methods for saving and loading objects."""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import pandas as pd

from py_metrics import caches

###########################################################################
# Estimator
###########################################################################

class Reg(object):
    def __init__(self, x, y):
        from scipy.linalg import lstsq
        self.lstsq = lstsq
        self.x = x
        self.y = y

        self._is_fit = False
        self.beta = None
        self.sse = None
        self.rnk_beta = None
        self.svs_beta = None

        self.qxx = None
        self.qxy = None

    def fit(self, frame):
        from numpy.linalg import inv

        # Prepare the data
        x = frame[self.x].copy(deep=True).astype(np.float32).values
        y = frame[self.y].copy(deep=True).astype(np.float32).values

        # other operations
        self.qxx = np.matmul(np.transpose(x), x)
        self.qxy = np.matmul(np.transpose(x), y)
        self.qxx_inv = inv(self.qxx)

        # Fit the regression
        self.beta, self.sse, self.rnk_beta, self.svs_beta = self.lstsq(x, y)
        self._is_fit = True

        # Compute summary stats
        self.ssy = ((y - y.mean()) ** 2).sum()
        self.omega_hat = self.see / frame.shape[0]
        self.r2 = (1 - self.sse / self.ssy)

    def predict(frame):
        x = frame[self.x].copy(deep=True).astype(np.float32).values
        return np.matmul(x, self.beta)


###########################################################################
# Covariance matrix estimation
###########################################################################

class AVar(object):
    def __init__(self):
        pass
