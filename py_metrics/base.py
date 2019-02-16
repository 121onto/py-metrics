# coding: utf-8

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
        self.k = len(x_cols)
        self.n = None

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
        self.e_bar = None
        self.sse = None
        self.ssy = None
        self.r2 = None

        # Error variance estimators
        self.o_hat = None
        self.o_til = None
        self.o_bar = None
        self.s_hat = None


    def fit(self, frame):
        # Prepare the data
        self.x = x = frame[self.x_cols].copy(deep=True).astype(np.float32).values
        self.y = y = frame[self.y_col].copy(deep=True).astype(np.float32).values
        self.n = frame.shape[0]
        n, k = self.n, self.k

        # other operations
        self.qxx = np.matmul(np.transpose(x), x)
        self.qxy = np.matmul(np.transpose(x), y)
        self.qxx_inv = inv(self.qxx)

        # Fit the regression
        self.beta, self.sse, _, _ = lstsq(x, y)
        self.ssy = ((y - y.mean()) ** 2).sum()
        self.r2 = (1 - self.sse / self.ssy)
        self._is_fit = True

        # Errors
        self.e_hat = e_hat = y - np.matmul(x, self.beta)
        self.e_til = ((1 - self.leverage()) ** -1 ) * e_hat
        self.e_bar = ((1 - self.leverage()) ** -0.5 ) * e_hat

        # Summary stats
        # TODO (121onto): run experiments to determine whether I should pull `leverage`
        #   out from under the `** 2` operator.
        self.o_hat = np.sqrt(
            self.sse / n)
        self.s_hat = np.sqrt(
            ((self.e_hat) ** 2).sum() / (n - k))
        self.o_til = np.sqrt(
            ((self.e_til) ** 2).sum() / n)
        self.o_bar = np.sqrt(
            ((self.e_bar) ** 2).sum() / n)


    def predict(self, frame):
        if not self._is_fit:
            raise RuntimeError('''
            You must run `Reg.fit` before running `Reg.predict`.''')

        x = frame[self.x_cols].copy(deep=True).astype(np.float32).values
        return np.matmul(x, self.beta)


    def acov(self, estimator='v_hc3'):
        if not self._is_fit:
            raise RuntimeError('''
            You must run `Reg.fit` before running `Reg.predict`.''')

        if estimator == 'v_0':
            return self.qxx_inv * (self.s_hat ** 2)
        elif estimator == 'v_hc0':
            e_hat, x = self.e_hat, self.x
            norm = 1
        elif estimator == 'v_hc1':
            e_hat, x = self.e_hat, self.x
            norm = self.n / (self.n - self.k)
        elif estimator == 'v_hc2':
            e_hat, x = self.e_bar, self.x
            norm = 1
        elif estimator == 'v_hc3':
            e_hat, x = self.e_til, self.x
            norm = 1

        one = three = self.qxx_inv
        two = np.matmul(
            np.multiply(np.transpose(x), e_hat),
            np.multiply(e_hat[:, np.newaxis], x)
        )
        acov = norm * np.matmul(np.matmul(one, two), three)
        return acov


    def nvar(self, estimator='norm_v0'):
        acov = self.acov(estimator=estimator)
        return np.diag(acov)


    def msfe(self):
        """Computes the mean-square forecast error (for a sample of size n-1).
        The forecast error is the difference between the actual value for y and
        it's point forecast. This is the forecast.  The mean-squared forecast
        error (MSFE) is its expected squared value.

        SOURCES:

            - Hansen, Chapter 4.121onto

        """
        if not self._is_fit:
            raise RuntimeError('''
            You must run `Reg.fit` before running `Reg.predict`.''')
        return self.o_til


    def leverage(self):
        """The leverage values hii measure how unusual the ith observation xi is
        relative to the other values in the sample. A large hii occurs when xi
        is quite different from the other sample values.

        SOURCES:

            - https://stackoverflow.com/a/39534036/759442
            - Hansen, Chapter 3.19

        """
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


    def influence(self):
        """Computes the largest (absolute) change in the predicted value due to
        a single observation. If this diag- nostic is large relative to the
        distribution of yi; it may indicate that that observation is ináuential.

        Observation i is ináuential if its omission from the sample induces
        a substantial change in a parameter estimate of interest.  Note that
        a leverage point is not necessarily ináuential as the latter also
        requires that the prediction error e is large.

        SOURCES:

            - Hansen, Chapter 3.21

        """
        return (self.h * self.e_til).max()


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
