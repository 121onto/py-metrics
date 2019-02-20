"""Test statistics.

Tests statistics including Wald...
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from scipy.linalg import lstsq
from scipy.stats import chi2

import numpy as np
import pandas as pd

###########################################################################
# Base class
###########################################################################

class Stat(object):
    def __init__(self, theta, theta_0=None, vce=None):
        raise NotImplementedError


    def fit(self):
        raise NotImplementedError


    def c_bound(self, alpha):
        raise NotImplementedError


    def p_value(self):
        raise NotImplementedError


###########################################################################
# TTest
###########################################################################

class Wald(Stat):
    def __init__(self, theta, theta_0=None, vce=None):
        """Student-t test statistic.

        Parameters
        ----------
        theta: np.array of type np.float32 and shape (q,)
            For example, you may use `Reg.beta` after running `Reg.fit()`
        theta_0: np.array of type np.float32 and shape (q,)
            The value of theta under the null hypothesis.  This parameter
            is optional and defaults to zero (optional).
        vce: np.array of type np.float32 and shape (q,q)
            An estimate of the variance-covariance matrix associated
            with theta.
        """
        if vce is None:
            raise ValueError('''
            `vce` must not be empty and should be an np.array of type
            np.float32 with shape (q,q) in call to initialize `test.Wald`.''')

        self.theta = theta
        self.theta_0 = np.zeros_like(theta) if theta_0 is None else theta_0
        self.vce = vce

        self.value = None
        self._is_fit = False
        self.fit()


    def fit(self):
        """Computes the Wald test statistic.
        """
        n = self.theta if self.theta_0 is None else (self.theta - self.theta_0)
        d = np.sqrt(np.diag(self.vce))

        self.value = (n / d)
        self._is_fit = True


    def c_bound(self, alpha):
        """Computes the confidence bound associated with the Wald test statistic.
        """
        return chi2.ppf(1 - alpha, df=self.q)


    def p_value(self):
        """Computes the p-value associated with the Wald test statistic.
        """
        if not self._is_fit:
            raise RuntimeError('''
            You must run `fit` before calling `p_value`.''')

        return chi2.cdf(self.value, df=self.q)


###########################################################################
# Wald
###########################################################################

class Wald(Stat):
    def __init__(self, theta, theta_0=None, vce=None):
        """Wald test statistic.

        Parameters
        ----------
        theta: np.array of type np.float32 and shape (q,)
            For example, you may use `Reg.beta` after running `Reg.fit()`
        theta_0: np.array of type np.float32 and shape (q,)
            The value of theta under the null hypothesis.  This parameter
            is optional and defaults to zero (optional).
        vce: np.array of type np.float32 and shape (q,q)
            An estimate of the variance-covariance matrix associated
            with theta.
        """
        if vce is None:
            raise ValueError('''
            `vce` must not be empty and should be an np.array of type
            np.float32 with shape (q,q) in call to initialize `test.Wald`.''')

        self.theta = theta
        self.theta_0 = np.zeros_like(theta) if theta_0 is None else theta_0
        self.vce = vce
        self.q = vce.shape[0]

        self.value = None
        self._is_fit = False
        self.fit()


    def fit(self):
        """Computes the Wald test statistic.
        """
        l = self.theta if self.theta_0 is None else (self.theta - self.theta_0)
        r, _, _, _ = lstsq(self.vce, l)

        self.value = np.matmul(np.transpose(l), r)
        self._is_fit = True


    def c_bound(self, alpha):
        """Computes the confidence bound associated with the Wald test statistic.
        """
        return chi2.ppf(1 - alpha, df=self.q)


    def p_value(self):
        """Computes the p-value associated with the Wald test statistic.
        """
        if not self._is_fit:
            raise RuntimeError('''
            You must run `fit` before calling `p_value`.''')

        return chi2.cdf(self.value, df=self.q)
