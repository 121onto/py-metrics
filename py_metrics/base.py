# coding: utf-8

"""Base classes."""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import pandas as pd

from scipy.linalg import lstsq
from scipy.stats import norm as normal
from scipy.stats import t as student_t
from numpy.linalg import inv
from numpy.linalg import cholesky

from py_metrics import caches

###########################################################################
# OLS
###########################################################################

class Reg(object):
    def __init__(self, x_cols, y_col):
        # TODO (121onto): enforce types, ensure everything gets defined

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

        # Error variance estimators
        self.o_hat = None
        self.o_til = None
        self.o_bar = None
        self.s_hat = None


    def residuals(self):
        if not self._is_fit:
            raise RuntimeError('''
            You must run `Reg.fit` before calling `Reg.residuals`.''')

        # Compute residuals
        x, y = self.x, self.y
        n, k = self.n, self.k

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
        self._is_fit = True

        self.residuals()


    def predict(self, frame):
        if not self._is_fit:
            raise RuntimeError('''
            You must run `fit` before calling `predict`.''')

        x = frame[self.x_cols].copy(deep=True).astype(np.float32).values
        return np.matmul(x, self.beta)


    def r2(self, estimator='r-til2'):
        """Compute R2 measure of fit.

        Parameters
        ----------
        estimator: string
            One of 'r2', 'r-bar2', or 'r-til2' (default: 'r-til2').

        Returns
        -------
        float

        Discussion
        ----------
        r2 is a commonly reported measure of regression fit.  One problem with
        r2; which is partially corrected by r-bar2 and fully corrected by r-til2;
        is that r2 necessarily increases when regressors are added to a
        regression model.  Another issue with r2 is that it uses biased
        estimators.  Note that r-til2 uses o_til which is used in computing the
        forecast error.  It is thus a much better measure of out-of-sample fit.

        r-til2 is popular for model comparison and selection, especially in high
        -dimensional (non- parametric) contexts.  Models with high r-til2 are
        better models in terms of expected out of sample squared error.
        In contrast, r2 cannot be used for model selection, as it necessarily
        increases when regressors are added to a regression model.

        Always prefer 'r-til2' over the alternatives.
        """
        if estimator not in ('r2', 'r-bar2', 'r-til2'):
            raise ValueError('''
            `estimator` must be one of 'r2', 'r-bar2', or 'r-til2'
            in call to `r2`.''')

        o_yhat = self.ssy / self.n
        if estimator == 'r2':
            r2 = (1 - (self.o_hat / o_yhat))
        elif estimator == 'r-bar2':
            r2 = (1 - (n-1) / (n-k) * (self.o_hat / o_yhat))
        elif estimator == 'r-til2':
            r2 = (1 - (self.o_til / o_yhat))
        else: # NOTE: this code should never execute
            r2 = None

        return r2


    def vce(self, estimator='hc2'):
        """Asymptotic covariance matrix estimation.

        Parameters
        ----------
        estimator: string
            One of '0', 'hc0', 'hc1', 'hc2', 'hc3'.

        Returns
        -------
        float

        Discussion
        ----------
        The four estimators HC0, HC1, HC2 and HC3 are collectively called
        robust, heteroskedasticity- consistent, or heteroskedasticity-robust
        covariance matrix estimators. The HC0 estimator was Örst developed
        by Eicker (1963) and introduced to econometrics by White (1980), and
        is sometimes called the Eicker-White or White covariance matrix
        estimator. The degree-of-freedom adjust- ment in HC1 was recommended
        by Hinkley (1977), and is the default robust covariance matrix estimator
        implemented in Stata. It is implement by the ì,rî option, for example
        by a regression executed with the command `reg y x, r`. In applied
        econometric practice, this is the currently most popular covariance
        matrix estimator. The HC2 estimator was introduced by Horn, Horn
        and Duncan (1975) (and is implemented using the vce(hc2) option
        in Stata). The HC3 estimator was derived by MacKinnon and White (1985)
        from the jackknife principle (see Section 10.3), and by Andrews (1991a)
        based on the principle of leave-one-out cross-validation (and is
        implemented using the vce(hc3) option in Stata).

        Which one to use?  `v_0` is typically a poor choice.  HC1 is the most
        commonly used as it is the default robust covariance matrix option in
        Stata. However, HC2 and HC3 are preferred. HC2 is unbiased (under
        homoskedasticity) and HC3 is conservative for any x. In most
        applications HC1, HC2 and HC3 will be very similar so this choice will
        not matter.  The context where the estimators can di§er substantially is
        when the sample has a large leverage value for some observation.

        The heteroskedasticity-robust covariance matrix estimators can be quite
        imprecise in some contexts. One is in the presence of sparse dummy
        variables ñwhen a dummy variable only takes the value 1 or 0 for very
        few observations.
        """
        if not self._is_fit:
            raise RuntimeError('''
            You must run `fit` before calling `vce`.''')

        if estimator not in ('0', 'hc0', 'hc1', 'hc2', 'hc3'):
            raise ValueError('''
            Argument `estimator` must be one of '0', 'hc0', 'hc1', 'hc2', or 'hc3'
            in call to `vce`.''')

        if estimator == '0':
            return self.qxx_inv * (self.s_hat ** 2)
        elif estimator == 'hc0':
            e_hat, x = self.e_hat, self.x
            norm = 1
        elif estimator == 'hc1':
            e_hat, x = self.e_hat, self.x
            norm = self.n / (self.n - self.k)
        elif estimator == 'hc2':
            e_hat, x = self.e_bar, self.x
            norm = 1
        elif estimator == 'hc3':
            e_hat, x = self.e_til, self.x
            norm = 1

        qxx_inv = self.qxx_inv
        omega = np.matmul(
            np.multiply(np.transpose(x), e_hat),
            np.multiply(e_hat[:, np.newaxis], x)
        )
        acov = norm * np.matmul(np.matmul(qxx_inv, omega), qxx_inv)
        return acov


    def ve(self, estimator='hc2'):
        return np.diag(self.vce(estimator=estimator))


    def std_err(self, estimator='hc2'):
        return np.sqrt(self.ve(estimator=estimator))


    def confidence_interval(self, alpha=0.05, dist='normal', vce='hc2'):
        """Compute a confidence interval with coverage 1 - alpha.

        Parameters
        ----------
        alpha: float
            1 - alpha equals the coverage probability.
        dist: string
            One of 'normal' or 'student-t'
        estimator: string
            One of '0', 'hc0', 'hc1', 'hc2', 'hc3'.

        Returns
        -------
        two np.arrays of type np.float32
        """
        if not self._is_fit:
            raise RuntimeError('''
            You must run `fit` before calling `confidence_interval`.''')

        if dist == 'normal':
            ppf = normal.ppf
        elif dist == 'student-t':
            ppf = lambda x: student_t.ppf(x, df=(self.n - self.k))
        else:
            raise ValueError('''
            Argument `dist` must be one of 'narmal' or 'student-t'
            in call to `confidence_interval`.''')

        c = ppf(1 - (alpha / 2))
        std_err = self.std_err(estimator=vce)
        lb = self.beta - c * std_err
        ub = self.beta + c * std_err
        return lb, ub


    def p_value(self, alpha=0.05, dist='normal', vce='hc2'):
        """Compute a confidence interval with coverage 1 - alpha.

        Parameters
        ----------
        alpha: float
            1 - alpha equals the coverage probability.
        dist: string
            One of 'normal' or 'student-t'
        estimator: string
            One of '0', 'hc0', 'hc1', 'hc2', 'hc3'.

        Returns
        -------
        two np.arrays of type np.float32
        """
        if not self._is_fit:
            raise RuntimeError('''
            You must run `fit` before calling `p_value`.''')

        if dist == 'normal':
            cdf = normal.cdf
        elif dist == 'student-t':
            cdf = lambda x: student_t.cdf(x, df=(self.n - self.k))
        else:
            raise ValueError('''
            Argument `dist` must be one of 'narmal' or 'student-t'
            in call to `p_value`.''')

        t = self.beta / self.std_err(estimator=vce)
        return 2 * (1 - cdf(np.absolute(t)))


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
            You must run `fit` before calling `msfe`.''')
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
            You must run `fit` before calling `leverage`.''')

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
        self.leverage()
        return (self.h * self.e_til).max()


    def summarize(self, alpha=0.05, dist='normal', vce='hc2'):
        if not self._is_fit:
            raise RuntimeError('''
            You must run `fit` before calling `summarize`.''')

        summary = pd.DataFrame(self.beta, index=self.x_cols, columns=['beta'])
        summary['se({})'.format(vce)] = self.std_err(estimator=vce)
        summary['p-value'] = self.p_value(dist=dist, vce=vce)

        print('='*80)
        print('y: {}'.format(self.y_col), '\n')
        print(summary, '\n')
        print('n: {}'.format(self.n))
        print('k: {}'.format(self.k))
        print('s_hat: {}'.format(self.s_hat))
        print('R2: {}'.format(self.r2()))


###########################################################################
# Cluster
###########################################################################

class Cluster(Reg):
    # Regression with cluster-robust inference.

    def __init__(self, x_cols, y_col, grp_col):
        self.grp_col = grp_col
        self.grp = None
        self.grp_idx = None

        super().__init__(x_cols, y_col)


    def residuals(self):
        if not self._is_fit:
            raise RuntimeError('''
            You must run `fit` before running `residuals`.''')

        # Compute residuals
        x, y = self.x, self.y
        n, k = self.n, self.k

        # Errors
        self.e_hat = e_hat = y - np.matmul(x, self.beta)
        self.e_til = np.zeros(e_hat.shape, dtype=np.float32)
        for grp, idx in self.grp_idx.items():
            A = (
                np.eye(len(idx)) -
                np.matmul(
                    np.matmul(x[idx], self.qxx_inv),
                    np.transpose(x[idx])))
            self.e_til[idx], _, _, _ = lstsq(A, e_hat[idx])
        self.e_bar = None # NOTE: not implemented

        # Summary stats
        self.o_hat = np.sqrt(
            self.sse / n)
        self.s_hat = np.sqrt(
            ((self.e_hat) ** 2).sum() / (n - k))
        self.o_til = np.sqrt(
            ((self.e_til) ** 2).sum() / n)
        self.o_bar = None # NOTE: not implemented


    def fit(self, frame):
        """Regression with cluster-robust inference.

        Discussion
        ----------
        There is a trade-off between bias and variance in the estimation of the
        covariance matrix by cluster-robust methods.

        First, suppose cluster dependence is ignored or imposed at too Öne a level
        (e.g. clustering by households instead of villages). Then variance
        estimators will be biased as they will omit covariance terms. As correlation
        is typically positive, this suggests that standard errors will be too small,
        giving rise to spurious indications of signiÖcance and precision.

        Second, suppose cluster dependence is imposed at too aggregate a measure
        (e.g. clustering by states rather than villages). This does not cause bias.
        But the variance estimators will contain many extra components, so the
        precision of the covariance matrix estimator will be poor. This means that
        reported standard errors will be imprecise ñmore random ñthan if
        clustering had been less aggregate.
        """
        self.grp = frame[self.grp_col].copy(deep=True).astype(np.float32).values
        self.grp_idx = frame.groupby([self.grp_col]).indices
        return super().fit(frame)


    def vce(self, estimator='cr3'):
        """Asymptotic covariance matrix estimation.

        Parameters
        ----------
        estimator: string
            One of 'cr0', 'cr1', 'cr3' (default is 'cr3').

        Returns
        -------
        float

        Discussion
        ----------
        The label CR refers to custer-robust and CR3 refers to the analogous
        formula for the HC3 esitmator.  Stata implements CR1 while CR3 is
        conservative.

        In many respects cluster-robust inference should be viewed similarly to
        heteroskedaticity-robust inference, where a cluster in the cluster-
        robust case is interpreted similarly to an observation in the
        heteroskedasticity-robust case. In particular, the effective sample size
        should be viewed as the number of clusters, not the sample size n.

        Most cluster-robust theory assumes that the clusters are
        homogeneous.  When this is violated ñwhen, for example, cluster sizes
        are highly heterogeneous ñthe regression should be viewed as roughly
        equivalent to the heteroskedasticity-robust case with an extremely high
        degree of heteroskedasticity.

        Put together, if the number of clusters G is small and the number of
        observations per cluster is highly varied, then we should interpret
        inferential statements with a great degree of caution.  Standard error
        estimates can be erroneously small. In the extreme of a single treated
        cluster (in the example, if only a single school was tracked) then the
        estimated coefficient on tracking will be very imprecisely estimated,
        yet will have a misleadingly small cluster standard error.
        """
        if not self._is_fit:
            raise RuntimeError('''
            You must run `fit` before calling `vce`.''')

        if estimator not in ('cr0', 'cr1', 'cr3'):
            raise ValueError('''
            Argument `estimator` must be one of 'cr0', 'cr1', or 'cr3'
            in call to `vce`.''')

        x, y, e_hat = self.x, self.y, self.e_hat
        n, k, G = self.n, self.k, len(self.grp_idx)

        if estimator == 'cr0':
            e_hat, x = self.e_hat, self.x
            norm = 1
        elif estimator == 'cr1':
            e_hat, x = self.e_hat, self.x
            norm = (n / (n - k)) * (G / (G - 1))
        elif estimator == 'cr3':
            e_hat, x = self.e_til, self.x
            norm = 1

        qxx_inv = self.qxx_inv
        omega = np.zeros([k, k], dtype=np.float32)
        for label, idx in self.grp_idx.items():
            vec = np.dot(e_hat[idx], x[idx,:])
            omega = omega + np.outer(vec, vec)
        acov = norm * np.matmul(np.matmul(qxx_inv, omega), qxx_inv)
        return acov


    def ve(self, estimator='cr3'):
        # see `Cluster.vce` for options
        return super().ve(estimator=estimator)


    def std_err(self, estimator='cr3'):
        # see `Cluster.vce` for options
        return super().std_err(estimator=estimator)


    def confidence_interval(self, alpha=0.05, dist='normal', vce='cr3'):
        """Compute a confidence interval with coverage 1 - alpha.

        Parameters
        ----------
        alpha: float
            1 - alpha equals the coverage probability.
        dist: string
            One of 'normal' or 'student-t' (default is 'normal').
        estimator: string
            One of 'cr0', 'cr1', or 'cr3' (default is 'cr3').

        Returns
        -------
        two np.arrays of type np.float32
        """
        return super().confidence_interval(alpha=alpha, dist=dist, vce=vce)


    def p_value(self, dist='normal', vce='cr3'):
        """Compute p-values

        Parameters
        ----------
        dist: string
            One of 'normal' or 'student-t' (default is 'normal').
        estimator: string
            One of 'cr0', 'cr1', or 'cr3' (default is 'cr3').

        Returns
        -------
        An np.arrays of type np.float32
        """
        return super().p_value(dist=dist, vce=vce)


    def leverage():
        raise NotImplementedError


    def influence():
        raise NotImplementedError


    def summarize(self, alpha=0.05, dist='normal', vce='cr3'):
        super().summarize(alpha=alpha, dist=dist, vce=vce)
