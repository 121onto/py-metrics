"""Regression in various forms."""
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

from py_metrics import core

###########################################################################
# OLS
###########################################################################

class Reg(object):
    def __init__(self, x_cols, y_col):
        # Columns defining estimation
        self.x_cols = x_cols
        self.y_col = y_col
        self.k = len(x_cols)
        self.n = None

        # Data arrays
        self.x = None
        self.y = None
        self.h = None

        # Flags
        self._is_fit = False

        # Hat matrices
        self.qxx = None
        self.qxx_inv = None

        # Coefficients and errors
        self.beta = None
        self.sse = None
        self.ssy = None

        # Residuals
        self.e_hat = None
        self.e_til = None
        self.e_bar = None

        # Residual variance estimators
        self.o_tot = None
        self.o_hat = None
        self.s_hat = None
        self.o_til = None
        self.o_bar = None


    def residuals(self):
        if not self._is_fit:
            raise RuntimeError('''
            You must run `Reg.fit` before calling `Reg.residuals`.''')

        # Data references
        x, y = self.x, self.y
        n, k = self.n, self.k

        # Residuals
        self.e_hat = e_hat = y - np.dot(x, self.beta)
        self.e_til = ((1 - self.leverage()) ** -1 ) * e_hat
        self.e_bar = ((1 - self.leverage()) ** -0.5 ) * e_hat

        # Residual variances
        self.o_tot = np.sqrt(
            (self.ssy / n))
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

        self.beta, self.sse = core._least_squares(x=x, y=y)
        self.ssy = core._ssy(y=y)
        self.qxx = qxx = core._sandwich(x=x)
        self.qxx_inv = inv(qxx)
        self._is_fit = True

        self.residuals()


    def predict(self, frame):
        if not self._is_fit:
            raise RuntimeError('''
            You must run `fit` before calling `predict`.''')

        x = frame[self.x_cols].copy(deep=True).astype(np.float32).values
        return np.dot(x, self.beta)


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

        SOURCE: Hansen, Chapter 4.18, page 125.
        """
        if estimator not in ('r2', 'r-bar2', 'r-til2'):
            raise ValueError('''
            `estimator` must be one of 'r2', 'r-bar2', or 'r-til2'
            in call to `r2`.''')

        n, k = self.n, self.k

        if estimator == 'r2':
            r2 = (1 - (self.o_hat / self.o_tot) ** 2)
        elif estimator == 'r-bar2':
            r2 = (1 - (n - 1) / (n - k) * (self.o_hat / self.o_tot) ** 2)
        elif estimator == 'r-til2':
            r2 = (1 - (self.o_til / self.o_tot) ** 2)
        else:
            raise NotImplementedError

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
        covariance matrix estimators. The HC0 estimator was first developed
        by Eicker (1963) and introduced to econometrics by White (1980), and
        is sometimes called the Eicker-White or White covariance matrix
        estimator. The degree-of-freedom adjust- ment in HC1 was recommended
        by Hinkley (1977), and is the default robust covariance matrix estimator
        implemented in Stata.  In econometric practice, this is the currently
        most popular covariance matrix estimator.

        The HC2 estimator was introduced by Horn, Horn
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
        not matter.  The context where the estimators can differ substantially is
        when the sample has a large leverage value for some observation.

        The heteroskedasticity-robust covariance matrix estimators can be quite
        imprecise in some contexts. One is in the presence of sparse dummy
        variables when a dummy variable only takes the value 1 or 0 for very
        few observations.

        SOURCE: Chapter 4.13 page 117; Chapter 4.14 pages 118-119.
        """
        if not self._is_fit:
            raise RuntimeError('''
            You must run `fit` before calling `vce`.''')

        if estimator not in ('0', 'hc0', 'hc1', 'hc2', 'hc3'):
            raise ValueError('''
            Argument `estimator` must be one of '0', 'hc0', 'hc1', 'hc2', or 'hc3'
            in call to `vce`.''')

        x = self.x
        n, k = self.n, self.k

        if estimator == '0':
            return self.qxx_inv * (self.s_hat ** 2)
        elif estimator == 'hc0':
            e_hat = self.e_hat
            norm = 1
        elif estimator == 'hc1':
            e_hat = self.e_hat
            norm = n / (n - k)
        elif estimator == 'hc2':
            e_hat = self.e_bar
            norm = 1
        elif estimator == 'hc3':
            e_hat = self.e_til
            norm = 1
        else:
            raise NotImplementedError

        xdx = core._sandwich(x=x, w=e_hat)
        return norm * core._sandwich(x=self.qxx_inv, w=xdx)


    def ve(self, estimator='hc2'):
        """Compute the variance associated with beta.

        Parameters
        ----------
        estimator: string
            One of '0', 'hc0', 'hc1', 'hc2', 'hc3'.

        Returns
        -------
        np.array of type np.float32
        """
        return np.diag(self.vce(estimator=estimator))


    def std_err(self, estimator='hc2'):
        """Compute the standard error associated with beta.

        Parameters
        ----------
        estimator: string
            One of '0', 'hc0', 'hc1', 'hc2', 'hc3'.

        Returns
        -------
        np.array of type np.float32
        """
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


    def p_value(self, dist='normal', vce='hc2'):
        """Computes a p-value for each regression coefficient.

        Parameters
        ----------
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
            raise NotImplementedError

        t = self.beta / self.std_err(estimator=vce)
        return 2 * (1 - cdf(np.absolute(t)))


    def msfe(self):
        """Computes the mean-square forecast error (for a sample of size n-1).
        The forecast error is the difference between the actual value for y and
        it's point forecast. This is the forecast.  The mean-squared forecast
        error (MSFE) is its expected squared value.

        SOURCE: Hansen, Chapter 4.12, page 115
        """
        if not self._is_fit:
            raise RuntimeError('''
            You must run `fit` before calling `msfe`.''')
        return self.o_til


    def leverage(self):
        """The leverage values hii measure how unusual the ith observation xi is
        relative to the other values in the sample. A large hii occurs when xi
        is quite different from the other sample values.

        SOURCE: Hansen, Chapter 3.19, page 88
            Also see: https://stackoverflow.com/a/39534036/759442
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
        distribution of yi; it may indicate that that observation is influential.

        Observation i is influential if its omission from the sample induces
        a substantial change in a parameter estimate of interest.  Note that
        a leverage point is not necessarily influential as the latter also
        requires that the prediction error e is large.

        SOURCE: Hansen, Chapter 3.21, page 91.
        """
        return (self.leverage() * self.e_til).max()


    def summarize(self, dist='normal', vce='hc2'):
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

        # Columns defining estimation
        self.x_cols = x_cols
        self.y_col = y_col
        self.grp_col = grp_col

        self.k = len(x_cols)
        self.n = None

        # Data arrays
        self.x = None
        self.y = None
        self.grp = None
        self.grp_idx = None

        # Flags
        self._is_fit = False

        # Hat matrices
        self.qxx = None
        self.qxx_inv = None

        # Coefficients and errors
        self.beta = None
        self.sse = None
        self.ssy = None

        # Residuals
        self.e_hat = None
        self.e_til = None

        # Residual variance estimators
        self.o_tot = None
        self.o_hat = None
        self.s_hat = None
        self.o_til = None


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
            lhs = np.eye(len(idx))
            rhs = core._sandwich(x=np.transpose(x[idx]), w=self.qxx_inv)
            self.e_til[idx], _, _, _ = lstsq(lhs - rhs, e_hat[idx])

        # Summary stats
        self.o_tot = np.sqrt(
            self.ssy / n)
        self.o_hat = np.sqrt(
            self.sse / n)
        self.s_hat = np.sqrt(
            ((self.e_hat) ** 2).sum() / (n - k))
        self.o_til = np.sqrt(
            ((self.e_til) ** 2).sum() / n)


    def fit(self, frame):
        """Regression with cluster-robust inference.

        discussion
        ----------
        There is a trade-off between bias and variance in the estimation of the
        covariance matrix by cluster-robust methods.

        First, suppose cluster dependence is ignored or imposed at too fine a level
        (e.g. clustering by households instead of villages). Then variance
        estimators will be biased as they will omit covariance terms. As correlation
        is typically positive, this suggests that standard errors will be too small,
        giving rise to spurious indications of significance and precision.

        Second, suppose cluster dependence is imposed at too aggregate a measure
        (e.g. clustering by states rather than villages). This does not cause bias.
        But the variance estimators will contain many extra components, so the
        precision of the covariance matrix estimator will be poor. This means that
        reported standard errors will be imprecise and more random than if
        clustering had been less aggregate.
        """

        # Cache group columns and indices
        self.grp = frame[self.grp_col].copy(deep=True).astype(np.float32).values
        self.grp_idx = frame.groupby([self.grp_col]).indices

        # Fit like you would in a normal regression context
        return super().fit(frame)


    def predict(self, frame):
        # see `Reg.predict` for details
        return super().predict(frame=frame)


    def r2(self, estimator='r-til2'):
        # TODO (121onto): confirm these calculations are reasonable in
        #   a clustered regression context.
        return super().r2(estimator=estimator)


    def vce(self, estimator='cr3'):
        """Cluster-robust asymptotic covariance matrix estimation.

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
        homogeneous.  When this is violated, for example, cluster sizes
        are highly heterogeneous, the regression should be viewed as roughly
        equivalent to the heteroskedasticity-robust case with an extremely high
        degree of heteroskedasticity.

        Put together, if the number of clusters G is small and the number of
        observations per cluster is highly varied, then we should interpret
        inferential statements with a great degree of caution.  Standard error
        estimates can be erroneously small. In the extreme of a single treated
        cluster (in the example, if only a single school was tracked) then the
        estimated coefficient on tracking will be very imprecisely estimated,
        yet will have a misleadingly small cluster standard error.

        SOURCE: Hansen, Chapter 4.21, page 133
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
            e_hat = self.e_hat
            norm = 1
        elif estimator == 'cr1':
            e_hat = self.e_hat
            norm = (n / (n - k)) * (G / (G - 1))
        elif estimator == 'cr3':
            e_hat = self.e_til
            norm = 1
        else:
            raise NotImplementedError

        qxx_inv = self.qxx_inv
        omega = np.zeros([k, k], dtype=np.float32)
        for label, idx in self.grp_idx.items():
            vec = np.dot(e_hat[idx], x[idx,:])
            omega = omega + np.outer(vec, vec)

        return norm * core._sandwich(x=qxx_inv, w=omega)


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
        """Computes a p-value for each regression coefficient.

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


    def msfe(self):
        raise NotImplementedError


    def leverage(self):
        raise NotImplementedError


    def influence(self):
        raise NotImplementedError


    def summarize(self, dist='normal', vce='cr3'):
        super().summarize(dist=dist, vce=vce)


###########################################################################
# Constrained least squares
###########################################################################

class CnsReg(Reg):
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

        # Flags
        self._is_fit = False

        # Hat matrices
        self.qxx = None
        self.qxx_inv = None
        self.qi_r = None
        self.r_qi_r = None

        # Coefficient estimates
        self.beta_ols = None
        self.beta = None
        self.sse_ols = None
        self.sse = None
        self.ssy = None

        # Diagnostics
        self.e_ols = None
        self.e_til = None

        # Error variance estimators
        self.o_tot = None
        self.o_ols = None
        self.s_ols = None
        self.o_til = None
        self.s_til = None


    def residuals(self):
        if not self._is_fit:
            raise RuntimeError('''
            You must run `fit` before calling `residuals`.''')

        # Compute residuals
        x, y = self.x, self.y
        n, k, q = self.n, self.k, self.q

        # residuals
        self.e_ols = e_ols = y - np.matmul(x, self.beta_ols)
        self.e_til = e_hat = y - np.matmul(x, self.beta)

        # errors
        self.sse_ols = sse_ols = (e_ols ** 2).sum()
        self.sse = sse = (e_til ** 2).sum()

        # se estimators
        self.o_tot = np.sqrt(ssy / n)
        self.o_ols = np.sqrt(sse_ols / n)
        self.s_ols = np.sqrt(sse_ols / (n - k))
        self.o_til = np.sqrt(sse / n)
        self.s_til = np.sqrt(sse / (n - k + q))


    def fit(self, frame, r=None, c=None):
        """Regression for constrained least squares.

        Parameters
        ----------
        frame: pd.DataFrame with columns of type np.float32
            The data.
        r: np.array of type np.float32 of shape (k,q)
            The constraint matrix (optional if set during initialization).
        c: np.array of type np.float32 of shape (q,)
            The constraint value such that R'beta = c (optional if set during initialization).

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

        self.beta_ols, _ = core._least_squares(x=x, y=y)
        self.ssy = core._ssy(y=y)
        self.qxx = core._sandwich(x=x)
        self.qxx_inv = inv(qxx)

        # cls operations
        self.qi_r = qi_r = np.matmul(self.qxx_inv, r)
        self.r_qi_r = r_qi_r = np.matmul(r, qi_r)
        rb = np.dot(np.transpose(r), self.beta_ols)
        rhs,_,_,_ = lstsq(r_qi_r, rb - c)
        lhs = qi_r

        # Fit the constrained regression
        self.beta = self.beta_ols - np.matmul(lhs, rhs)
        self._is_fit = True

        self.residuals()


    def predict(self, frame):
        # see `Reg.predict` for details
        return super().predict(frame=frame)


    def r2(self, estimator='r-bar2'):
        """Compute R2 measure of fit.

        Parameters
        ----------
        estimator: string
            Must equal one of 'r2' or 'r-bar2' (defaults to 'r-bar2').

        Returns
        -------
        float

        Discussion
        ----------
        These are ad-hoc at this point.

        TODO (121onto): confirm these calculations are reasonable.
        """
        if estimator not in ('r2', 'r-bar2'):
            raise ValueError('''
            `estimator` must be one of 'r2' or 'r-bar2' in call to `r2`.''')

        x, y = self.x, self.y
        n, k, q = self.n, self.k, self.q

        if estimator == 'r2':
            o_hat = self.o_ols
            r2 = (1 - (o_hat / o_tot) ** 2)
        elif estimator == 'r-bar2':
            o_hat = self.o_til
            r2 = (1 - (n - 1) / (n - k + q) * (o_hat / o_tot) ** 2)
        else:
            raise NotImplementedError

        return r2


    def vce(self, estimator='cns2'):
        """Constrained least squares asymptotic covariance matrix estimation.

        Parameters
        ----------
        estimator: string
            One of '0', 'cns0', 'cns1', or 'cns2' (defautl 'cns2').

        Returns
        -------
        float

        Discussion
        ----------
        The covariance estimator '0' works under the assumption of homoskedasticity.
        Both 'cns0' and 'cns1' are heteroskedasticity-robust and based on residuals
        from the unconstrained regression.  'cns2' uses residuals from the
        constrained regression.

        SOURCES: Homoskedastic estimator is from Hansen, Chapter 8.4, page 263.
        The heteroskedasticity-robust estimator is from Hansen, Chapter 8.7,
        page 268.
        """
        if not self._is_fit:
            raise RuntimeError('''
            You must run `fit` before calling `vce`.''')

        if estimator not in ('0', 'cns0', 'cns1', 'cns2'):
            raise ValueError('''
            Argument `estimator` must be one of '0', 'cns0', 'cns1', 'cns2'
            in call to `vce`.''')

        x, r = self.x, self.r
        n, k, q = self.n, self.k, self.q

        if estimator == '0':
            o_hat = self.s_til
            rhs,_,_,_ = lstsq(self.r_qi_r, np.transpose(self.qi_r))
            vce = (self.qxx_inv - np.matmul(self.qi_r, rhs)) * (o_hat ** 2)
            return vce
        elif estimator == 'cns0':
            e_hat = self.e_ols
            norm = 1
        elif estimator == 'cns1':
            e_hat = self.e_ols
            norm = n / (n - k)
        elif estimator == 'cns2':
            e_hat = self.e_til
            norm = n / (n - k + q)
        else:
            raise NotImplementedError

        qxx_inv, qi_r, r_qi_r = self.qxx_inv, self.qi_r, self.r_qi_r
        omega= core._sandwich(x=x, w=e_hat)
        v_beta = norm * core._sandwich(x=qxx_inv, w=omega)
        solve_r,_,_,_ = lstsq(r_qi_r, np.transpose(r))

        lhs = np.matmul(qi_r, solve_r)
        rhs = np.matmul(np.matmul(r, solve_r), qxx_inv)
        vce = (
            v_beta
            - np.matmul(lhs, v_beta)
            - np.matmul(v_beta, rhs)
            + np.matmul(np.matmul(lhs, v_beta), rhs))

        return vce


    def ve(self, estimator='cns2'):
        # see `CnsReg.vce` for options
        return super().ve(estimator=estimator)


    def std_err(self, estimator='cns2'):
        # see `CnsReg.vce` for options
        return super().std_err(estimator=estimator)


    def confidence_interval(self, alpha=0.05, dist='normal', vce='cns2'):
        """Compute a confidence interval with coverage 1 - alpha.

        Parameters
        ----------
        alpha: float
            1 - alpha equals the coverage probability.
        dist: string
            One of 'normal' or 'student-t'
        estimator: string
            One of '0', ...

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
            ppf = lambda x: student_t.ppf(x, df=(self.n - self.k + self.q))
        else:
            raise ValueError('''
            Argument `dist` must be one of 'narmal' or 'student-t'
            in call to `confidence_interval`.''')

        c = ppf(1 - (alpha / 2))
        std_err = self.std_err(estimator=vce)
        lb = self.beta - c * std_err
        ub = self.beta + c * std_err
        return lb, ub


    def p_value(self, dist='normal', vce='cns2'):
        """Computes a p-value for each regression coefficient.

        Parameters
        ----------
        dist: string
            One of 'normal' or 'student-t'
        estimator: string
            One of '0', ...

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
            cdf = lambda x: student_t.cdf(x, df=(self.n - self.k + self.q))
        else:
            raise ValueError('''
            Argument `dist` must be one of 'narmal' or 'student-t'
            in call to `p_value`.''')

        t = self.beta / self.std_err(estimator=vce)
        return 2 * (1 - cdf(np.absolute(t)))


    def msfe(self):
        raise NotImplementedError


    def leverage(self):
        raise NotImplementedError


    def influence(self):
        raise NotImplementedError


    def summarize(self, dist='normal', vce='cns2'):
        super().summarize(dist=dist, vce=vce)
