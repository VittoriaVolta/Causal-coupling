#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 15:09:03 2020

@author: vittoriavolta
"""

import os, sys, cmath, math, warnings, datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, optimize, linalg, special 
from scipy.interpolate import splrep, splev
from copy import deepcopy
import statsmodels.api as sm
from dateutil.relativedelta import relativedelta
from numpy import ma, atleast_2d, pi, sqrt, sum, transpose
from scipy.special import gammaln, logsumexp
from scipy.stats.mstats import mquantiles


from mpl_toolkits.mplot3d import axes3d
import matplotlib.cm as cm   
from pyhht.utils import extr, boundary_conditions
warnings.filterwarnings("ignore")

directory = '/Users/vittoriavolta/Desktop/article'

announc = pd.read_excel("announcements.xlsx")

for c in announc.columns:
    announc[c] = pd.to_datetime(announc[c]).dt.date 
    
ecb = announc["ECB"].dropna()
boe = announc["BoE"].dropna()

data_ecb = pd.DataFrame()
data_boe = pd.DataFrame()
for filename in os.listdir(directory):
    if filename.startswith("EST") and filename.endswith(".txt"):
        dataset = pd.read_csv(filename, header = None)
        dataset.columns = ["datetime", "open", "high", "low", "close"]
        dataset[['date','time']] = dataset.datetime.str.split(" ", expand=True) 
        dataset.drop("datetime", axis = 1, inplace = True)
        dataset["date"] = pd.to_datetime(dataset["date"]).dt.date 
        dataset["time"] = pd.to_datetime(dataset["time"]).dt.time 
        data_ecb = pd.concat([data_ecb, dataset])
    elif filename.startswith("UKX") and filename.endswith(".txt"):
        dataset = pd.read_csv(filename, header = None)
        dataset.columns = ["datetime", "open", "high", "low", "close"]
        dataset[['date','time']] = dataset.datetime.str.split(" ", expand=True) 
        dataset.drop("datetime", axis = 1, inplace = True)
        dataset["date"] = pd.to_datetime(dataset["date"]).dt.date 
        dataset["time"] = pd.to_datetime(dataset["time"]).dt.time 
        data_boe = pd.concat([data_boe, dataset])
        
    
# ECB

startTime_ecb = datetime.time(9,1)
endTime_ecb = datetime.time(17,29)
strDate_ecb = datetime.date(2015,12,10)
endDate_ecb = data_ecb["date"].max()


businessDates_ecb = pd.date_range(strDate_ecb, endDate_ecb, freq = "B").date
businessHours_ecb = pd.date_range(str(startTime_ecb), str(endTime_ecb), freq = "1min").time

format_data_ecb = data_ecb.pivot_table(index = "time", columns = "date", values = "close")
format_data_ecb = format_data_ecb.loc[:, format_data_ecb.columns.isin(businessDates_ecb)]
format_data_ecb = format_data_ecb.loc[format_data_ecb.index.isin(businessHours_ecb),:]
format_data_ecb = format_data_ecb.loc[:, format_data_ecb.columns[format_data_ecb.isnull().mean() < 0.3]]

sqr_rets_ecb = (np.log(format_data_ecb) - np.log(format_data_ecb.shift(1)))**2
ann = (format_data_ecb.shape[0]-1)*252
vol_ecb = np.sqrt(sqr_rets_ecb.rolling(window = 5).mean())*np.sqrt(ann)
vol_est_ecb = vol_ecb.loc[:, vol_ecb.columns.isin(ecb)]
vol_est_boe = vol_ecb.loc[:, vol_ecb.columns.isin(boe)]

# BoE

startTime_boe = datetime.time(3,0)
endTime_boe = datetime.time(11,29)
strDate_boe = datetime.date(2015,1,1)
endDate_boe = data_boe["date"].max()


businessDates_boe = pd.date_range(strDate_boe, endDate_boe, freq = "B").date
businessHours_boe = pd.date_range(str(startTime_boe), str(endTime_boe), freq = "1min").time

format_data_boe = data_boe.pivot_table(index = "time", columns = "date", values = "close")
format_data_boe = format_data_boe.loc[:, format_data_boe.columns.isin(businessDates_boe)]
format_data_boe = format_data_boe.loc[format_data_boe.index.isin(businessHours_boe),:]
format_data_boe = format_data_boe.loc[:, format_data_boe.columns[format_data_boe.isnull().mean() < 0.3]]

sqr_rets_boe = (np.log(format_data_boe) - np.log(format_data_boe.shift(1)))**2
ann = (format_data_boe.shape[0]-1)*252
vol_boe = np.sqrt(sqr_rets_boe.rolling(window = 5).mean())*np.sqrt(ann)
vol_ukx_boe = vol_boe.loc[:, vol_boe.columns.isin(boe)]
vol_ukx_ecb = vol_boe.loc[:, vol_boe.columns.isin(ecb)]


vol_est_ecb.reset_index(inplace = True)
time_ts = vol_est_ecb["time"]
vol_est_ecb.drop("time", axis = 1, inplace = True)

vol_ukx_ecb.reset_index(inplace = True)
vol_ukx_ecb.drop("time", axis = 1, inplace = True)

#vol_est_boe.reset_index(inplace = True)
#time_ts = vol_est_boe["time"]
#vol_est_boe.drop("time", axis = 1, inplace = True)

#vol_ukx_boe.reset_index(inplace = True)
#vol_ukx_boe.drop("time", axis = 1, inplace = True)





class EmpiricalModeDecomposition(object):
    """The EMD class."""

    def __init__(self, x, t=None, threshold_1=0.05, threshold_2=0.5,
                 alpha=0.05, ndirs=4, fixe=0, maxiter=2000, fixe_h=0, n_imfs=7,
                 nbsym=2, bivariate_mode='bbox_center'):
        """Empirical mode decomposition.
        Parameters
        ----------
        x : array-like, shape (n_samples,)
            The signal on which to perform EMD
        t : array-like, shape (n_samples,), optional
            The timestamps of the signal.
        threshold_1 : float, optional
            Threshold for the stopping criterion, corresponding to
            :math:`\\theta_{1}` in [3]. Defaults to 0.05.
        threshold_2 : float, optional
            Threshold for the stopping criterion, corresponding to
            :math:`\\theta_{2}` in [3]. Defaults to 0.5.
        alpha : float, optional
            Tolerance for the stopping criterion, corresponding to
            :math:`\\alpha` in [3]. Defaults to 0.05.
        ndirs : int, optional
            Number of directions in which interpolants for envelopes are
            computed for bivariate EMD. Defaults to 4. This is ignored if the
            signal is real valued.
        fixe : int, optional
            Number of sifting iterations to perform for each IMF. By default,
            the stopping criterion mentioned in [1] is used. If set to a
            positive integer, each mode is either the result of exactly
            `fixe` number of sifting iterations, or until a pure IMF is
            found, whichever is sooner.
        maxiter : int, optional
            Upper limit of the number of sifting iterations for each mode.
            Defaults to 2000.
        n_imfs : int, optional
            Number of IMFs to extract. By default, this is ignored and
            decomposition is continued until a monotonic trend is left in the
            residue.
        nbsym : int, optional
            Number of extrema to use to mirror the signals on each side of
            their boundaries.
        bivariate_mode : str, optional
            The algorithm to be used for bivariate EMD as described in [4].
            Can be one of 'centroid' or 'bbox_center'. This is ignored if the
            signal is real valued.
        Attributes
        ----------
        is_bivariate : bool
            Whether the decomposer performs bivariate EMD. This is
            automatically determined by the input value. This is True if at
            least one non-zero imaginary component is found in the signal.
        nbits : list
            List of number of sifting iterations it took to extract each IMF.
        References
        ----------
        .. [1] Huang H. et al. 1998 'The empirical mode decomposition and the \
                Hilbert spectrum for nonlinear and non-stationary time series \
                analysis.' \
                Procedings of the Royal Society 454, 903-995
        .. [2] Zhao J., Huang D. 2001 'Mirror extending and circular spline \
                function for empirical mode decomposition method'. \
                Journal of Zhejiang University (Science) V.2, No.3, 247-252
        .. [3] Gabriel Rilling, Patrick Flandrin, Paulo Gonçalves, June 2003: \
                'On Empirical Mode Decomposition and its Algorithms',\
                IEEE-EURASIP Workshop on Nonlinear Signal and Image Processing \
                NSIP-03
        .. [4] Gabriel Rilling, Patrick Flandrin, Paulo Gonçalves, \
                Jonathan M. Lilly. Bivariate Empirical Mode Decomposition. \
                10 pages, 3 figures. Submitted to Signal Processing Letters, \
                IEEE. Matlab/C codes and additional .. 2007. <ensl-00137611>
        Examples
        --------
        >>> from pyhht.visualization import plot_imfs
        >>> import numpy as np
        >>> t = np.linspace(0, 1, 1000)
        >>> modes = np.sin(2 * pi * 5 * t) + np.sin(2 * pi * 10 * t)
        >>> x = modes + t
        >>> decomposer = EMD(x)
        >>> imfs = decomposer.decompose()
        >>> plot_imfs(x, imfs, t) #doctest: +SKIP
        .. plot:: ../../docs/examples/simple_emd.py
        """

        self.threshold_1 = threshold_1
        self.threshold_2 = threshold_2
        self.alpha = alpha
        self.maxiter = maxiter
        self.fixe_h = fixe_h
        self.ndirs = ndirs
        self.nbit = 0
        self.Nbit = 0
        self.n_imfs = n_imfs
        self.k = 1
        # self.mask = mask
        self.nbsym = nbsym
        self.nbit = 0
        self.NbIt = 0

        if x.ndim > 1:
            if 1 not in x.shape:
                raise ValueError("x must have only one row or one column.")
        if x.shape[0] > 1:
            x = x.ravel()
        if np.any(np.isinf(x)):
            raise ValueError("All elements of x must be finite.")
        self.x = x
        self.ner = self.nzr = len(self.x)
        self.residue = self.x.copy()

        if t is None:
            self.t = np.arange(max(x.shape))
        else:
            if t.shape != self.x.shape:
                raise ValueError("t must have the same dimensions as x.")
            if t.ndim > 1:
                if 1 not in t.shape:
                    raise ValueError("t must have only one column or one row.")
            if not np.all(np.isreal(t)):
                raise TypeError("t must be a real vector.")
            if t.shape[0] > 1:
                t = t.ravel()
            self.t = t

        if fixe:
            self.maxiter = fixe
            if self.fixe_h:
                raise TypeError("Cannot use both fixe and fixe_h modes")
        self.fixe = fixe

        self.is_bivariate = np.any(np.iscomplex(self.x))
        if self.is_bivariate:
            self.bivariate_mode = bivariate_mode

        self.imf = []
        self.nbits = []
        

    def io(self):
        r"""Compute the index of orthoginality, as defined by:
        .. math::
            \sum_{i,j=1,i\neq j}^{N}\frac{\|C_{i}\overline{C_{j}}\|}{\|x\|^2}
        Where :math:`C_{i}` is the :math:`i` th IMF.
        Returns
        -------
        float
            Index of orthogonality. Lower values are better.
        Examples
        --------
        >>> import numpy as np
        >>> t = np.linspace(0, 1, 1000)
        >>> modes = np.sin(2 * pi * 5 * t) + np.sin(2 * pi * 10 * t)
        >>> x = modes + t
        >>> decomposer = EMD(x)
        >>> imfs = decomposer.decompose()
        >>> print('%.3f' % decomposer.io())
        0.017
        """
        imf = np.array(self.imf)
        dp = np.dot(imf, np.conj(imf).T)
        mask = np.logical_not(np.eye(len(self.imf)))
        s = np.abs(dp[mask]).sum()
        return s / (2 * np.sum(self.x ** 2))
      
        
    def stop_EMD(self):
        """Check if there are enough extrema (3) to continue sifting.
        Returns
        -------
        bool
            Whether to stop further cubic spline interpolation for lack of
            local extrema.
        """
        if self.is_bivariate:
            stop = False
            for k in range(self.ndirs):
                phi = k * math.pi / self.ndirs
                indmin, indmax, _ = extr(
                    np.real(np.exp(1j * phi) * self.residue))
                if len(indmin) + len(indmax) < 3:
                    stop = True
                    break
        else:
            indmin, indmax, _ = extr(self.residue)
            ner = len(indmin) + len(indmax)
            stop = ner < 3
        return stop

    def mean_and_amplitude(self, m):
        """ Compute the mean of the envelopes and the mode amplitudes.
        Parameters
        ----------
        m : array-like, shape (n_samples,)
            The input array or an itermediate value of the sifting process.
        Returns
        -------
        tuple
            A tuple containing the mean of the envelopes, the number of
            extrema, the number of zero crosssing and the estimate of the
            amplitude of themode.
        """
        # FIXME: The spline interpolation may not be identical with the MATLAB
        # implementation. Needs further investigation.
        if self.is_bivariate:
            if self.bivariate_mode == 'centroid':
                nem = []
                nzm = []
                envmin = np.zeros((self.ndirs, len(self.t)))
                envmax = np.zeros((self.ndirs, len(self.t)))
                for k in range(self.ndirs):
                    phi = k * math.pi / self.ndirs
                    y = np.real(np.exp(-1j * phi) * m)
                    indmin, indmax, indzer = extr(y)
                    nem.append(len(indmin) + len(indmax))
                    nzm.append(len(indzer))
                    if self.nbsym:
                        tmin, tmax, zmin, zmax = boundary_conditions(
                            y, self.t, m, self.nbsym)
                    else:
                        tmin = np.r_[self.t[0], self.t[indmin], self.t[-1]]
                        tmax = np.r_[self.t[0], self.t[indmax], self.t[-1]]
                        zmin, zmax = m[tmin], m[tmax]

                    f = splrep(tmin, zmin)
                    spl = splev(self.t, f)
                    envmin[k, :] = spl

                    f = splrep(tmax, zmax)
                    spl = splev(self.t, f)
                    envmax[k, :] = spl

                envmoy = np.mean((envmin + envmax) / 2, axis=0)
                amp = np.mean(abs(envmax - envmin), axis=0) / 2

            elif self.bivariate_mode == 'bbox_center':
                nem = []
                nzm = []
                envmin = np.zeros((self.ndirs, len(self.t)), dtype=complex)
                envmax = np.zeros((self.ndirs, len(self.t)), dtype=complex)
                for k in range(self.ndirs):
                    phi = k * math.pi / self.ndirs
                    y = np.real(np.exp(-1j * phi) * m)
                    indmin, indmax, indzer = extr(y)
                    nem.append(len(indmin) + len(indmax))
                    nzm.append(len(indzer))
                    if self.nbsym:
                        tmin, tmax, zmin, zmax = boundary_conditions(
                            y, self.t, m, self.nbsym)
                    else:
                        tmin = np.r_[self.t[0], self.t[indmin], self.t[-1]]
                        tmax = np.r_[self.t[0], self.t[indmax], self.t[-1]]
                        zmin, zmax = m[tmin], m[tmax]
                    f = splrep(tmin, zmin)
                    spl = splev(self.t, f)
                    envmin[k, ] = np.exp(1j * phi) * spl

                    f = splrep(tmax, zmax)
                    spl = splev(self.t, f)
                    envmax[k, ] = np.exp(1j * phi) * spl

                envmoy = np.mean((envmin + envmax), axis=0)
                amp = np.mean(abs(envmax - envmin), axis=0) / 2

        else:
            indmin, indmax, indzer = extr(m)
            nem = len(indmin) + len(indmax)
            nzm = len(indzer)
            if self.nbsym:
                tmin, tmax, mmin, mmax = boundary_conditions(m, self.t, m,
                                                             self.nbsym)
            else:
                tmin = np.r_[self.t[0], self.t[indmin], self.t[-1]]
                tmax = np.r_[self.t[0], self.t[indmax], self.t[-1]]
                mmin, mmax = m[tmin], m[tmax]

            f = splrep(tmin, mmin)
            envmin = splev(self.t, f)

            f = splrep(tmax, mmax)
            envmax = splev(self.t, f)

            envmoy = (envmin + envmax) / 2
            amp = np.abs(envmax - envmin) / 2.0
        if self.is_bivariate:
            nem = np.array(nem)
            nzm = np.array(nzm)

        return envmoy, nem, nzm, amp

    def stop_sifting(self, m):
        """Evaluate the stopping criteria for the current mode.
        Parameters
        ----------
        m : array-like, shape (n_samples,)
            The current mode.
        Returns
        -------
        bool
            Whether to stop sifting. If this evaluates to true, the current
            mode is interpreted as an IMF.
        """
        # FIXME: This method needs a better name.
        if self.fixe:
            (moyenne, _, _, _), stop_sift = self.mean_and_amplitude(m), 0  # NOQA
        elif self.fixe_h:
            stop_count = 0
            try:
                moyenne, nem, nzm = self.mean_and_amplitude(m)[:3]

                if np.all(abs(nzm - nem) > 1):
                    stop = 0
                    stop_count = 0
                else:
                    stop_count += 1
                    stop = (stop_count == self.fixe_h)
            except:
                moyenne = np.zeros((len(m)))
                stop = 1
            stop_sift = stop
        else:
            try:
                envmoy, nem, nzm, amp = self.mean_and_amplitude(m)
            except TypeError as err:
                if err.args[0] == "m > k must hold":
                    return 1, np.zeros((len(m)))
            except ValueError as err:
                if err.args[0] == "Not enough extrema.":
                    return 1, np.zeros((len(m)))
            sx = np.abs(envmoy) / amp
            stop = not(((np.mean(sx > self.threshold_1) > self.alpha) or
                        np.any(sx > self.threshold_2)) and np.all(nem > 2))
            if not self.is_bivariate:
                stop = stop and not(np.abs(nzm - nem) > 1)
            stop_sift = stop
            moyenne = envmoy
        return stop_sift, moyenne

    def keep_decomposing(self):
        """Check whether to continue the sifting operation."""
        return not(self.stop_EMD()) and \
            (self.k < self.n_imfs + 1 or self.n_imfs == 0)  # and \
# not(np.any(self.mask))

    def decompose(self):
        """Decompose the input signal into IMFs.
        This function does all the heavy lifting required for sifting, and
        should ideally be the only public method of this class.
        Returns
        -------
        imfs : array-like, shape (n_imfs, n_samples)
            A matrix containing one IMF per row.
        Examples
        --------
        >>> from pyhht.visualization import plot_imfs
        >>> import numpy as np
        >>> t = np.linspace(0, 1, 1000)
        >>> modes = np.sin(2 * pi * 5 * t) + np.sin(2 * pi * 10 * t)
        >>> x = modes + t
        >>> decomposer = EMD(x)
        >>> imfs = decomposer.decompose()
        """
        while self.keep_decomposing():

            # current mode
            m = self.residue

            # computing mean and stopping criterion
            stop_sift, moyenne = self.stop_sifting(m)

            # in case current mode is small enough to cause spurious extrema
            if np.max(np.abs(m)) < (1e-10) * np.max(np.abs(self.x)):
                if not stop_sift:
                    warnings.warn(
                        "EMD Warning: Amplitude too small, stopping.")
                else:
                    print("Force stopping EMD: amplitude too small.")
                return

            # SIFTING LOOP:
            while not(stop_sift) and (self.nbit < self.maxiter):
                # The following should be controlled by a verbosity parameter.
                # if (not(self.is_bivariate) and
                #     (self.nbit > self.maxiter / 5) and
                #     self.nbit % np.floor(self.maxiter / 10) == 0 and
                #     not(self.fixe) and self.nbit > 100):
                #     print("Mode " + str(self.k) +
                #           ", Iteration " + str(self.nbit))
                #     im, iM, _ = extr(m)
                #     print(str(np.sum(m[im] > 0)) + " minima > 0; " +
                #           str(np.sum(m[im] < 0)) + " maxima < 0.")

                # Sifting
                m = m - moyenne

                # Computing mean and stopping criterion
                stop_sift, moyenne = self.stop_sifting(m)

                self.nbit += 1
                self.NbIt += 1

                # This following warning depends on verbosity and needs better
                # handling
                # if not self.fixe and self.nbit > 100(self.nbit ==
                # (self.maxiter - 1)) and not(self.fixe) and (self.nbit > 100):
                #     warnings.warn("Emd:warning, Forced stop of sifting - " +
                #                   "Maximum iteration limit reached.")

            self.imf.append(m)

            self.nbits.append(self.nbit)
            self.nbit = 0
            self.k += 1

            self.residue = self.residue - m
            self.ort = self.io()

        if np.any(self.residue):
            self.imf.append(self.residue)
        return np.array(self.imf)




###########################
###### FUNCTIONS ##########
###########################


def sanitise(df):
    """
        Function to convert DataFrame-like objects into pandas DataFrames
        
    Args:
        df          -        Data in pd.Series or pd.DataFrame format
    Returns:
        df          -        Data as pandas DataFrame
    """
    ## Ensure data is in DataFrame form
    if isinstance(df, pd.DataFrame):
        df = df
    elif isinstance(df, pd.Series):
        df = df.to_frame()
    else:
        raise ValueError('Data passed as %s Please ensure your data is stored as a Pandas DataFrame' %(str(type(df))) )
    return df



def shuffle_series(DF, only=None):
    """
    Function to return time series shuffled rowwise along each desired column. 
    Each column is shuffled independently, removing the temporal relationship.
    This is to calculate Z-score and Z*-score. See P. Boba et al (2015)
    Calculated using:       df.apply(np.random.permutation)
    Arguments:
        df              -   (DataFrame) Time series data 
        only            -   (list)      Fieldnames to shuffle. If none, all columns shuffled 
    Returns:
        df_shuffled     -   (DataFrame) Time series shuffled along desired columns    
    """
    if not only == None:
        shuffled_DF = DF.copy()
        for col in only:
            series = DF.loc[:, col].to_frame()
            shuffled_DF[col] = series.apply(np.random.permutation)
    else:
        shuffled_DF = DF.apply(np.random.permutation)
    
    return shuffled_DF


         
def significance(df, TE, endog, exog, lag, n_shuffles, method, pdf_estimator=None, bins=None, bandwidth=None,  both=True):
        """
        Perform significance analysis on the hypothesis test of statistical causality, for both X(t)->Y(t)
        and Y(t)->X(t) directions
   
        Calculated using:  Assuming stationarity, we shuffle the time series to provide the null hypothesis. 
                           The proportion of tests where TE > TE_shuffled gives the p-value significance level.
                           The amount by which the calculated TE is greater than the average shuffled TE, divided
                           by the standard deviation of the results, is the z-score significance level.
        Arguments:
            TE              -      (list)    Contains the transfer entropy in each direction, i.e. [TE_XY, TE_YX]
            endog           -      (string)  The endogenous variable in the TE analysis being significance tested (i.e. X or Y) 
            exog            -      (string)  The exogenous variable in the TE analysis being significance tested (i.e. X or Y) 
            pdf_estimator   -      (string)  The pdf_estimator used in the original TE analysis
            bins            -      (Dict of lists)  The bins used in the original TE analysis
            n_shuffles      -      (float) Number of times to shuffle the dataframe, destroyig temporality
            both            -      (Bool) Whether to shuffle both endog and exog variables (z-score) or just exog                                  variables (giving z*-score)  
        Returns:
            p_value         -      Probablity of observing the result given the null hypothesis
            z_score         -      Number of Standard Deviations result is from mean (normalised)
        """ 

        ## Prepare array for Transfer Entropy of each Shuffle
        shuffled_TEs = np.zeros(shape = (2,n_shuffles))
        
        ##
        if both is True:
            pass #TBC

        for i in range(n_shuffles):
                ## Perform Shuffle
                df = shuffle_series(df)
                
                ## Calculate New TE
                shuffled_causality = TransferEntropy(   DF = df,
                                                endog = endog,     
                                                exog = exog,          
                                                lag = lag
                                            )    
                if method == 'linear':
                    TE_shuffled = shuffled_causality.linear_TE(df, n_shuffles=0)
                else:       
                    TE_shuffled = shuffled_causality.nonlinear_TE(df, pdf_estimator, bins, bandwidth, n_shuffles=0)
                shuffled_TEs[:,i] = TE_shuffled

        
        ## Calculate p-values for each direction
        p_values = (np.count_nonzero(TE[0] < shuffled_TEs[0,:]) /n_shuffles , \
                    np.count_nonzero(TE[1] < shuffled_TEs[1,:]) /n_shuffles)

        ## Calculate z-scores for each direction
        z_scores = ( ( TE[0] - np.mean(shuffled_TEs[0,:]) ) / np.std(shuffled_TEs[0,:]) , \
                     ( TE[1] - np.mean(shuffled_TEs[1,:]) ) / np.std(shuffled_TEs[1,:])  )
        
        TE_mean = ( np.mean(shuffled_TEs[0,:]), \
                     np.mean(shuffled_TEs[1,:]) )
        
        ## Return the self.DF value to the unshuffled case
        return p_values, z_scores, TE_mean


def get_entropy(df, gridpoints=15, bandwidth=None, estimator='kernel', bins=None, covar=None):
    """
        Function for calculating entropy from a probability mass 
        
    Args:
        df          -       (DataFrame) Samples over which to estimate density
        gridpoints  -       (int)       Number of gridpoints when integrating KDE over 
                                        the domain. Used if estimator='kernel'
        bandwidth   -       (float)     Bandwidth for KDE (scalar multiple to covariance
                                        matrix). Used if estimator='kernel'
        estimator   -       (string)    'histogram' or 'kernel'
        bins        -       (Dict of lists) Bin edges for NDHistogram. Used if estimator
                                        = 'histogram'
        covar       -       (Numpy ndarray) Covariance matrix between dimensions of df. 
                                        Used if estimator = 'kernel'
    Returns:
        entropy     -       (float)     Shannon entropy in bits
    """
    pdf = get_pdf(df, gridpoints, bandwidth, estimator, bins, covar)
    ## log base 2 returns H(X) in bits
    return -np.sum( pdf * ma.log2(pdf).filled(0))



def get_pdf(df, gridpoints=None, bandwidth=None, estimator=None, bins=None, covar=None):
    """
        Function for non-parametric density estimation
    Args:
        df          -       (DataFrame) Samples over which to estimate density
        gridpoints  -       (int)       Number of gridpoints when integrating KDE over 
                                        the domain. Used if estimator='kernel'
        bandwidth   -       (float)     Bandwidth for KDE (scalar multiple to covariance
                                        matrix). Used if estimator='kernel'
        estimator   -       (string)    'histogram' or 'kernel'
        bins        -       (Dict of lists) Bin edges for NDHistogram. Used if estimator = 'histogram'
        covar       -       (Numpy ndarray) Covariance matrix between dimensions of df. 
                                        Used if estimator = 'kernel'
    Returns:
        pdf         -       (Numpy ndarray) Probability of a sample being in a specific 
                                        bin (technically a probability mass)
    """
    DF = sanitise(df)
    
    if estimator == 'histogram':
        pdf = pdf_histogram(DF, bins)
 #   else:
 #       pdf = pdf_kde(DF, gridpoints, bandwidth, covar)
    return pdf



def pdf_histogram(df,bins):
    """
        Function for non-parametric density estimation using N-Dimensional Histograms
    Args:
        df            -       (DataFrame) Samples over which to estimate density
        bins          -       (Dict of lists) Bin edges for NDHistogram. 
    Returns:
        histogram.pdf -       (Numpy ndarray) Probability of a sample being in a specific 
                                    bin (technically a probability mass)
    """
    histogram = NDHistogram(df=df, bins=bins)        
    return histogram.pdf



#############################
######### CLASS #############
#############################


class LaggedTimeSeries():
    """
        Custom wrapper class for pandas DataFrames for performing predictive analysis.
        Generates lagged time series and performs custom windowing over datetime indexes
    """
    def __init__(self, df, lag=None, max_lag_only=True, window_size = None, window_stride = None):
        """
        Args:
            df              -   Pandas DataFrame object of N columns. Must be indexed as an increasing 
                                time series (i.e. past-to-future), with equal timesteps between each row
            lags            -   The number of steps to be included. Each increase in Lags will result 
                                in N additional columns, where N is the number of columns in the original 
                                dataframe. It will also remove the first N rows.
            max_lag_only    -   Defines whether the returned dataframe contains all lagged timeseries up to 
                                and including the defined lag, or only the time series equal to this lag value
            window_size     -   Dict containing key-value pairs only from within: {'YS':0,'MS':0,'D':0,'H':0,'min':0,'S':0,'ms':0}
                                Describes the desired size of each window, provided the data is indexed with datetime type. Leave as
                                None for no windowing. Units follow http://pandas.pydata.org/pandas-docs/stable/timeseries.html#timeseries-offset-aliases
            window_stride   -   Dict containing key-value pairs only from within: {'YS':0,'MS':0,'D':0,'H':0,'min':0,'S':0,'ms':0}
                                Describes the size of the step between consecutive windows, provided the data is indexed with datetime type. Leave as
                                None for no windowing. Units follow http://pandas.pydata.org/pandas-docs/stable/timeseries.html#timeseries-offset-aliases
                       
        Returns:    -   n/a
        """        
        
        self.df = sanitise(df)
        self.axes = list(self.df.columns.values) #Variable names

        self.max_lag_only = max_lag_only
        if lag is not None:
            self.t = lag
            self.df = self.__apply_lags__()

        if window_size is not None and window_stride is not None:
            self.has_windows = True
            self. __apply_windows__(window_size, window_stride)
        else:
            self.has_windows = False       
        

    def __apply_lags__(self):
        """
        Args:
            n/a
        Returns:
            new_df.iloc[self.t:]    -   This is a new dataframe containing the original columns and
                                        all lagged columns. Note that the first few rows (equal to self.lag) will
                                        be removed from the top, since lagged values are of coursenot available
                                        for these indexes.
        """
        # Create a new dataframe to maintain the new data, dropping rows with NaN
        new_df = self.df.copy(deep=True).dropna()

        # Create new column with lagged timeseries for each variable
        col_names = self.df.columns.values.tolist()

        # If the user wants to only consider the time series lagged by the 
        # maximum number specified or by every series up to an including the maximum lag:
        if self.max_lag_only == True:
            for col_name in col_names:
                new_df[col_name + '_lag' + str(self.t)] = self.df[col_name].shift(self.t)

        elif self.max_lag_only == False:
            for col_name in col_names:
                for t in range(1,self.t+1):
                    new_df[col_name + '_lag' + str(t)] = self.df[col_name].shift(t)
        else:
            raise ValueError('Error')

        # Drop the first t rows, which now contain NaN
        return new_df.iloc[self.t:]

    def __apply_windows__(self, window_size, window_stride):
        """
        Args:
            window_size      -   Dict passed from self.__init__
            window_stride    -   Dict passed from self.__init__
        Returns:    
            n/a              -   Sets the daterange for the self.windows property to iterate along
        """
        self.window_size =  {'YS':0,'MS':0,'D':0,'H':0,'min':0,'S':0,'ms':0}
        self.window_stride =  {'YS':0,'MS':0,'D':0,'H':0,'min':0,'S':0,'ms':0}

        self.window_stride.update(window_stride)
        self.window_size.update(window_size)
        freq = ''
        daterangefreq = freq.join([str(v)+str(k) for (k,v) in self.window_stride.items() if v != 0])
        self.daterange = pd.date_range(self.df.index.min(),self.df.index.max() , freq=daterangefreq)

    def date_diff(self,window_size):
        """
        Args: 
            window_size     -    Dict passed from self.windows function
        Returns:
            start_date      -    The start date of the proposed window
            end_date        -    The end date of the proposed window    
        
        This function is TBC - proposed due to possible duplication of the relativedelta usage in self.windows and self.headstart
        """
        pass

    @property
    def windows(self):
        """
        Args: 
            n/a
        Returns:
            windows         -   Generator defining a pandas DataFrame for each window of the data. 
                                Usage like:   [window for window in LaggedTimeSeries.windows]
        """
        if self.has_windows == False:
            return self.df
        ## Loop Over TimeSeries Range
        for i,dt in enumerate(self.daterange):
            
            ## Ensure Each Division Contains Required Number of Months
            if dt-relativedelta(years   =  self.window_size['YS'],
                                months  =  self.window_size['MS'],
                                days    =  self.window_size['D'],
                                hours   =  self.window_size['H'],
                                minutes =  self.window_size['min'],
                                seconds =  self.window_size['S'],
                                microseconds = self.window_size['ms']
                                ) >= self.df.index.min():
                
                ## Create Window 
                yield self.df.loc[(dt-relativedelta(years   =  self.window_size['YS'],
                                                    months  =  self.window_size['MS'],
                                                    days    =  self.window_size['D'],
                                                    hours   =  self.window_size['H'],
                                                    minutes =  self.window_size['min'],
                                                    seconds =  self.window_size['S'],
                                                    microseconds = self.window_size['ms']
                                                    )) : dt]

    @property
    def headstart(self):
        """
        Args: 
            n/a
        Returns:
            len(windows)    -   The number of windows which would have start dates before the desired date range. 
                                Used in TransferEntropy class to slice off incomplete windows.
            
        """
        windows =   [i for i,dt in enumerate(self.daterange) 
                            if dt-relativedelta(    years   =  self.window_size['YS'],
                                                    months  =  self.window_size['MS'],
                                                    days    =  self.window_size['D'],
                                                    hours   =  self.window_size['H'],
                                                    minutes =  self.window_size['min'],
                                                    seconds =  self.window_size['S'],
                                                    microseconds = self.window_size['ms']
                                        ) < self.df.index.min() ]
        ## i.e. count from the first window which falls entirely after the earliest date
        return len(windows)




class TransferEntropy():
    """
        Functional class to calculate Transfer Entropy between time series, to detect causal signals.
        Currently accepts two series: X(t) and Y(t). Future extensions planned to accept additional endogenous 
        series: X1(t), X2(t), X3(t) etc. 
    """

    def __init__(self, DF, endog, exog, lag = None, window_size=None, window_stride=None):
        """
        Args:
            DF            -   (DataFrame) Time series data for X and Y (NOT including lagged variables)
            endog         -   (string)    Fieldname for endogenous (dependent) variable Y
            exog          -   (string)    Fieldname for exogenous (independent) variable X
            lag           -   (integer)   Number of periods (rows) by which to lag timeseries data
            window_size   -   (Dict)      Must contain key-value pairs only from within: {'YS':0,'MS':0,'D':0,'H':0,'min':0,'S':0,'ms':0}
                                          Describes the desired size of each window, provided the data is indexed with datetime type. Leave as
                                          None for no windowing. Units follow http://pandas.pydata.org/pandas-docs/stable/timeseries.html#timeseries-offset-aliases
            window_stride -   (Dict)      Must contain key-value pairs only from within: {'YS':0,'MS':0,'D':0,'H':0,'min':0,'S':0,'ms':0}
                                          Describes the size of the step between consecutive windows, provided the data is indexed with datetime type. Leave as
                                          None for no windowing. Units follow http://pandas.pydata.org/pandas-docs/stable/timeseries.html#timeseries-offset-aliases
        Returns:
            n/a
        """
        
        self.lts = LaggedTimeSeries(df=sanitise(DF), 
                                    lag=lag, 
                                    window_size=window_size,
                                    window_stride=window_stride)
        

        if self.lts.has_windows is True:
            self.df = self.lts.windows
            self.date_index = self.lts.daterange[self.lts.headstart:]
            self.results = pd.DataFrame(index=self.date_index)
            self.results.index.name = "windows_ending_on"
        else:
            self.df = [self.lts.df]
            self.results = pd.DataFrame(index=[0])
        self.max_lag_only = True
        self.endog = endog                             # Dependent Variable Y
        self.exog = exog                               # Independent Variable X
        self.lag = lag
        
        
    def linear_TE(self, df=None, n_shuffles=0):
        """
        Linear Transfer Entropy for directional causal inference
        Defined:            G-causality * 0.5, where G-causality described by the reduction in variance of the residuals
                            when considering side information.
        Calculated using:   log(var(e_joint)) - log(var(e_independent)) where e_joint and e_independent
                            represent the residuals from OLS fitting in the joint (X(t),Y(t)) and reduced (Y(t)) cases
        Arguments:
            n_shuffles  -   (integer)   Number of times to shuffle the dataframe, destroying the time series temporality, in order to 
                                        perform significance testing.
        Returns:
            transfer_entropies  -  (list) Directional Linear Transfer Entropies from X(t)->Y(t) and Y(t)->X(t) respectively
        """
        ## Prepare lists for storing results
        TEs = []
        shuffled_TEs = []
        p_values = []
        z_scores = []

        ## Loop over all windows
        for i,df in enumerate(self.df):
            df = deepcopy(df)

            ## Shows user that something is happening
            if self.lts.has_windows is True:
                print("Window ending: ", self.date_index[i])

            ## Initialise list to return TEs
            transfer_entropies = [0,0]

        
            ## Require us to compare information transfer bidirectionally
            for i,(X,Y) in enumerate({self.exog:self.endog, self.endog:self.exog}.items()):

                ## Note X-t, Y-t
                X_lagged = X+'_lag'+str(self.lag)
                Y_lagged = Y+'_lag'+str(self.lag)

                ## Calculate Residuals after OLS Fitting, for both Independent and Joint Cases
                joint_residuals = sm.OLS(df[Y], sm.add_constant(df[[Y_lagged,X_lagged]])).fit().resid
                independent_residuals = sm.OLS(df[Y], sm.add_constant(df[Y_lagged])).fit().resid 

                ## Use Geweke's formula for Granger Causality 
                granger_causality = np.log(    np.var(independent_residuals) /
                                np.var(joint_residuals))
                
                ## Calculate Linear Transfer Entropy from Granger Causality
                transfer_entropies[i] = granger_causality/2

            TEs.append(transfer_entropies)
        

            ## Calculate Significance of TE during this window
            if n_shuffles > 0:
                p, z, TE_mean = significance(df = df, 
                                        TE = transfer_entropies, 
                                        endog = self.endog, 
                                        exog = self.exog, 
                                        lag = self.lag,
                                        n_shuffles = n_shuffles,
                                        method='linear')

                shuffled_TEs.append(TE_mean)
                p_values.append(p)
                z_scores.append(z)


        ## Store Linear Transfer Entropy from X(t)->Y(t) and from Y(t)->X(t)
        self.add_results({'TE_linear_XY' : np.array(TEs)[:,0],
                          'TE_linear_YX' : np.array(TEs)[:,1],
                          'p_value_linear_XY' : None,
                          'p_value_linear_YX' : None,
                          'z_score_linear_XY' : 0,
                          'z_score_linear_YX' : 0
                          })

        if n_shuffles > 0:
            ## Store Significance Transfer Entropy from X(t)->Y(t) and from Y(t)->X(t)
            
            self.add_results({'p_value_linear_XY' : np.array(p_values)[:,0],
                              'p_value_linear_YX' : np.array(p_values)[:,1],
                              'z_score_linear_XY' : np.array(z_scores)[:,0],
                              'z_score_linear_YX' : np.array(z_scores)[:,1],
                              'Ave_TE_linear_XY'  : np.array(shuffled_TEs)[:,0],
                              'Ave_TE_linear_YX'  : np.array(shuffled_TEs)[:,1]
                              })

        return transfer_entropies    



    def nonlinear_TE(self, df=None, pdf_estimator='histogram', bins=None, bandwidth=None, gridpoints=20, n_shuffles=0):
        """
        NonLinear Transfer Entropy for directional causal inference
        Defined:            TE = TE_XY - TE_YX      where TE_XY = H(Y|Y-t) - H(Y|Y-t,X-t)
        Calculated using:   H(Y|Y-t,X-t) = H(Y,Y-t,X-t) - H(Y,Y-t)  and finding joint entropy through density estimation
        Arguments:
            pdf_estimator   -   (string)    'Histogram' or 'kernel' Used to define which method is preferred for density estimation
                                            of the distribution - either histogram or KDE
            bins            -   (dict of lists) Optional parameter to provide hard-coded bin-edges. Dict keys 
                                            must contain names of variables - including lagged columns! Dict values must be lists
                                            containing bin-edge numerical values. 
            bandwidth       -   (float)     Optional parameter for custom bandwidth in KDE. This is a scalar multiplier to the covariance
                                            matrix used (see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.covariance_factor.html)
            gridpoints      -   (integer)   Number of gridpoints (in each dimension) to discretise the probablity space when performing
                                            integration of the kernel density estimate. Increasing this gives more precision, but significantly
                                            increases execution time
            n_shuffles      -   (integer)   Number of times to shuffle the dataframe, destroying the time series temporality, in order to 
                                            perform significance testing.
        Returns:
            transfer_entropies  -  (list) Directional Transfer Entropies from X(t)->Y(t) and Y(t)->X(t) respectively
        
        (Also stores TE, Z-score and p-values in self.results - for each window if windows defined.)
        """


        ## Retrieve user-defined bins
        self.bins = bins
        if self.bins is None:
            self.bins = {self.endog: None}

        ## Prepare lists for storing results
        TEs = []
        shuffled_TEs = []
        p_values = []
        z_scores = []

        ## Loop over all windows
        for i,df in enumerate(self.df):
            df = deepcopy(df)

            ## Shows user that something is happening
            if self.lts.has_windows is True:
                print("Window ending: ", self.date_index[i])

            ## Initialise list to return TEs
            transfer_entropies = [0,0]

            ## Require us to compare information transfer bidirectionally
            for i,(X,Y) in enumerate({self.exog:self.endog, self.endog:self.exog}.items()):
                
                ### Entropy calculated using Probability Density Estimation:
                    # Following: https://stat.ethz.ch/education/semesters/SS_2006/CompStat/sk-ch2.pdf
                    # Also: https://www.cs.cmu.edu/~aarti/Class/10704_Spring15/lecs/lec5.pdf
                
                ## Note Lagged Terms
                X_lagged = X+'_lag'+str(self.lag)
                Y_lagged = Y+'_lag'+str(self.lag)

                ### Estimate PDF using Gaussian Kernels and use H(x) = p(x) log p(x)

                ## 1. H(Y,Y-t,X-t)  
                H1 = get_entropy(df = df[[Y,Y_lagged,X_lagged]], 
                                gridpoints = gridpoints,
                                bandwidth = bandwidth, 
                                estimator = pdf_estimator,
                                bins = {k:v for (k,v) in self.bins.items()
                                        if k in[Y,Y_lagged,X_lagged]})
                ## 2. H(Y-t,X-t)
                H2 = get_entropy(df = df[[X_lagged,Y_lagged]],
                                gridpoints = gridpoints,
                                bandwidth = bandwidth,
                                estimator = pdf_estimator,
                                bins = {k:v for (k,v) in self.bins.items() 
                                        if k in [X_lagged,Y_lagged]}) 
                ## 3. H(Y,Y-t)  
                H3 = get_entropy(df = df[[Y,Y_lagged]],
                                gridpoints = gridpoints,
                                bandwidth  = bandwidth,
                                estimator = pdf_estimator,
                                bins =  {k:v for (k,v) in self.bins.items() 
                                        if k in [Y,Y_lagged]})
                ## 4. H(Y-t)  
                H4 = get_entropy(df = df[[Y_lagged]],
                                gridpoints = gridpoints,
                                bandwidth  = bandwidth,
                                estimator = pdf_estimator,
                                bins =  {k:v for (k,v) in self.bins.items() 
                                        if k in [Y_lagged]})                


                ### Calculate Conditonal Entropy using: H(Y|X-t,Y-t) = H(Y,X-t,Y-t) - H(X-t,Y-t)
                conditional_entropy_joint =  H1 - H2
            
                ### And Conditional Entropy independent of X(t) H(Y|Y-t) = H(Y,Y-t) - H(Y-t)            
                conditional_entropy_independent = H3 - H4

                ### Directional Transfer Entropy is the difference between the conditional entropies
                transfer_entropies[i] =  conditional_entropy_independent - conditional_entropy_joint
            
            TEs.append(transfer_entropies)

            ## Calculate Significance of TE during this window
            if n_shuffles > 0:
                p, z, TE_mean = significance(    df = df, 
                                        TE = transfer_entropies, 
                                        endog = self.endog, 
                                        exog = self.exog, 
                                        lag = self.lag, 
                                        n_shuffles = n_shuffles, 
                                        pdf_estimator = pdf_estimator, 
                                        bins = self.bins,
                                        bandwidth = bandwidth,
                                        method='nonlinear')

                shuffled_TEs.append(TE_mean)
                p_values.append(p)
                z_scores.append(z)

        ## Store Transfer Entropy from X(t)->Y(t) and from Y(t)->X(t)
        self.add_results({'TE_XY' : np.array(TEs)[:,0],
                          'TE_YX' : np.array(TEs)[:,1],
                          'p_value_XY' : None,
                          'p_value_YX' : None,
                          'z_score_XY' : 0,
                          'z_score_YX' : 0
                          })
        if n_shuffles > 0:
            ## Store Significance Transfer Entropy from X(t)->Y(t) and from Y(t)->X(t)
            
            self.add_results({'p_value_XY' : np.array(p_values)[:,0],
                              'p_value_YX' : np.array(p_values)[:,1],
                              'z_score_XY' : np.array(z_scores)[:,0],
                              'z_score_YX' : np.array(z_scores)[:,1],
                              'Ave_TE_XY'  : np.array(shuffled_TEs)[:,0],
                              'Ave_TE_YX'  : np.array(shuffled_TEs)[:,1]
                            })

        return transfer_entropies

    

    def add_results(self,dict):
        """
        Args:
            dict    -   JSON-style data to store in existing self.results DataFrame
        Returns:
            n/a
        """
        for (k,v) in dict.items():
            self.results[str(k)] = v   
            
            
            

class NDHistogram():
    """
        Custom histogram class wrapping the default numpy implementations (np.histogram, np.histogramdd). 
        This allows for dimension-agnostic histogram calculations, custom auto-binning and 
        associated data and methods to be stored for each object (e.g. Probability Density etc.)
    """
    def __init__(self, df, bins=None, max_bins = 15):
        """
        Arguments:
            df          -   DataFrame passed through from the TransferEntropy class
            bins        -   Bin edges passed through from the TransferEntropy class
            max_bins    -   Number of bins per each dimension passed through from the TransferEntropy class
        Returns:
            self.pdf    -   This is an N-dimensional Probability Density Function, stored as a
                            Numpy histogram, representing the proportion of samples in each bin.
        """
        df = sanitise(df)            
        self.df = df.reindex(columns= sorted(df.columns))   # Sort axes by name
        self.max_bins = max_bins
        self.axes = list(self.df.columns.values)
        self.bins = bins
        self.n_dims = len(self.axes)

        ## Bins must match number and order of dimensions
        if self.bins is None:
            AB = AutoBins(self.df)
            self.bins = AB.sigma_bins(max_bins=max_bins)
        elif set(self.bins.keys()) != set(self.axes):
            warnings.warn('Incompatible bins provided - defaulting to sigma bins')
            AB = AutoBins(self.df)
            self.bins = AB.sigma_bins(max_bins=max_bins)
            
        ordered_bins = [sorted(self.bins[key]) for key in sorted(self.bins.keys())]

        ## Create ND histogram (np.histogramdd doesn't scale down to 1D)
        if self.n_dims == 1:
            self.Hist, self.Dedges = np.histogram(self.df.values,bins=ordered_bins[0], normed=False)
        elif self.n_dims > 1:
            self.Hist, self.Dedges = np.histogramdd(self.df.values,bins=ordered_bins, normed=False)
        

        ## Empirical Probability Density Function
        if self.Hist.sum() == 0:   
            print(self.Hist.shape)
            
            with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
                print(self.df.tail(40))

            sys.exit("User-defined histogram is empty. Check bins or increase data points")
        else:
            self.pdf = self.Hist/self.Hist.sum()
            self._set_entropy_(self.pdf)
  
    def _set_entropy_(self,pdf):
        """
        Arguments:
            pdf   -   Probabiiity Density Function; this is calculated using the N-dimensional histogram above.
        Returns:
            n/a   
        Sets entropy for marginal distributions: H(X), H(Y) etc. as well as joint entropy H(X,Y)
        """
        ## Prepare empty dict for marginal entropies along each dimension
        self.H = {}

        if self.n_dims >1:
            
            ## Joint entropy H(X,Y) = -sum(pdf(x,y) * log(pdf(x,y)))     
            self.H_joint =  -np.sum(pdf * ma.log2(pdf).filled(0)) # Use masking to replace log(0) with 0

            ## Single entropy for each dimension H(X) = -sum(pdf(x) * log(pdf(x)))
            for a, axis_name in enumerate(self.axes):
                self.H[axis_name] =  -np.sum(pdf.sum(axis=a) * ma.log2(pdf.sum(axis=a)).filled(0)) # Use masking to replace log(0) with 0
        else:
            ## Joint entropy and single entropy are the same
            self.H_joint = -np.sum(pdf * ma.log2(pdf).filled(0)) 
            self.H[self.df.columns[0]] = self.H_joint            
                    
            
            
            
class AutoBins():
    """
        Prototyping class for generating data-driven binning.
        Handles lagged time series, so only DF[X(t), Y(t)] required.
    """
    def __init__(self, df, lag=None):
        """
        Args:
            df      -   (DateFrame) Time series data to classify into bins
            lag     -   (float)     Lag for data to provided bins for lagged columns also
        Returns:
            n/a
        """
        ## Ensure data is in DataFrame form
        self.df = sanitise(df)
        self.axes = self.df.columns.values
        self.ndims = len(self.axes)
        self.N = len(self.df)
        self.lag = lag

    def __extend_bins__(self, bins):
        """
           Function to generate bins for lagged time series not present in self.df
        Args:   
            bins    -   (Dict of List)  Bins edges calculated by some AutoBins.method()
        Returns:
            bins    -   (Dict of lists) Bin edges keyed by column name
        """
        self.max_lag_only = True # still temporary until we kill this

        ## Handle lagging for bins, and calculate default bins where edges are not provided
        if self.max_lag_only == True:
            bins.update({   fieldname + '_lag' + str(self.lag): edges   
                            for (fieldname,edges) in bins.items()})  
        else:
            bins.update({   fieldname + '_lag' + str(t): edges          
                            for (fieldname,edges) in bins.items() for t in range(self.lag)})
        
        return bins

    def MIC_bins(self, max_bins):
        """
        Method to find optimal bin widths in each dimension, using a naive search to 
        maximise the mutual information divided by number of bins. Only accepts data
        with two dimensions [X(t),Y(t)].
        We increase the n_bins parameter in each dimension, and take the bins which
        result in the greatest Maximum Information Coefficient (MIC)
        
        (Note that this is restricted to equal-width bins only.)
        Defined:            MIC = I(X,Y)/ max(n_bins)
                            edges = {Y:[a,b,c,d], Y-t:[a,b,c,d], X-t:[e,f,g]}, 
                            n_bins = [bx,by]
        Calculated using:   argmax { I(X,Y)/ max(n_bins) }
        Args:
            max_bins        -   (int)       The maximum allowed bins in each dimension
        Returns:
            opt_edges       -   (dict)      The optimal bin-edges for pdf estimation
                                            using the histogram method, keyed by df column names
                                            All bins equal-width.
        """
        if len(self.df.columns.values) > 2:
            raise ValueError('Too many columns provided in DataFrame. MIC_bins only accepts 2 columns (no lagged columns)')

  
        quants = np.linspace(0,1,max_bins)
        
        bins = {}
        
        for i,dim in enumerate(self.df.columns.values):
            arr_quants = []
            for qu in quants:
                val = np.quantile(self.df[dim], qu)
                arr_quants.append(val)
            bins[dim] = arr_quants
        
        
        if self.lag is not None:
            bins = self.__extend_bins__(bins)
        ## Return the optimal bin-edges
        
        return bins

    def knuth_bins(self,max_bins=15):
        """
        Method to find optimal bin widths in each dimension, using a naive search to 
        maximise the log-likelihood given data. Only accepts data
        with two dimensions [X(t),Y(t)]. 
        Derived from Matlab code provided in Knuth (2013):  https://arxiv.org/pdf/physics/0605197.pdf
        
        (Note that this is restricted to equal-width bins only.)
        Args:
            max_bins        -   (int)       The maximum allowed bins in each dimension
        Returns:
            bins            -   (dict)      The optimal bin-edges for pdf estimation
                                            using the histogram method, keyed by df column names
                                            All bins equal-width.
        """
        if len(self.df.columns.values) > 2:
            raise ValueError('Too many columns provided in DataFrame. knuth_bins only accepts 2 columns (no lagged columns)')

        
        min_bins = 3

        ## Initialise array to store MIC values
        log_probabilities = np.zeros(shape=[1+max_bins-min_bins,1+max_bins-min_bins])
        
        ## Loop over each dimension 
        for b_x in range(min_bins, max_bins+1):

            for b_y in range(min_bins, max_bins+1):
                
                ## Update parameters
                Ms = [b_x,b_y]
                
                ## Update dict of bin edges
                bins = {dim :  list(np.linspace(    self.df[dim].min(), 
                                                    self.df[dim].max(), 
                                                    int(Ms[i]+1)))
                                for i,dim in enumerate(self.df.columns.values)}

                ## Calculate Maximum log Posterior
                
                # Create N-d histogram to count number per bin
                HDE = NDHistogram(self.df, bins)
                nk = HDE.Hist

                # M = number of bins in total =  Mx * My * Mz ... etc.
                M = np.prod(Ms)

                log_prob = ( self.N * np.log(M)
                            + gammaln(0.5 * M)
                            - M * gammaln(0.5)
                            - gammaln(self.N + 0.5 * M)
                            + np.sum(gammaln(nk.ravel() + 0.5)))

                log_probabilities[b_x-min_bins][b_y-min_bins] = log_prob 
        

        ## Get Optimal b_x, b_y values
        Ms[0] = np.where(log_probabilities == np.max(log_probabilities))[0] + min_bins
        Ms[1] = np.where(log_probabilities == np.max(log_probabilities))[1] + min_bins
        
        bins = {dim :  list(   np.linspace(self.df[dim].min(), 
                                                self.df[dim].max(), 
                                                int(Ms[i]+1)))
                            for i,dim in enumerate(self.df.columns.values)}
        
        if self.lag is not None:
            bins = self.__extend_bins__(bins)
        ## Return the optimal bin-edges
        return bins

    def sigma_bins(self, max_bins=15):
        """ 
        Returns bins for N-dimensional data, using standard deviation binning: each 
        bin is one S.D in width, with bins centered on the mean. Where outliers exist 
        beyond the maximum number of SDs dictated by the max_bins parameter, the
        bins are extended to minimum/maximum values to ensure all data points are
        captured. This may mean larger bins in the tails, and up to two bins 
        greater than the max_bins parameter suggests in total (in the unlikely case of huge
        outliers on both sides). 
        Args:
            max_bins        -   (int)       The maximum allowed bins in each dimension
        Returns:
            bins            -   (dict)      The optimal bin-edges for pdf estimation
                                            using the histogram method, keyed by df column names
        """

        
        bins = {k:[np.mean(v)-int(max_bins/2)*np.std(v) + i * np.std(v) for i in range(max_bins+1)] 
                for (k,v) in self.df.iteritems()}   # Note: same as:  self.df.to_dict('list').items()}

        # Since some outliers can be missed, extend bins if any points are not yet captured
        [bins[k].append(self.df[k].min()) for k in self.df.keys() if self.df[k].min() < min(bins[k])]
        [bins[k].append(self.df[k].max()) for k in self.df.keys() if self.df[k].max() > max(bins[k])]

        if self.lag is not None:
            bins = self.__extend_bins__(bins)
        return bins

    def equiprobable_bins(self,max_bins=15):
        """ 
        Returns bins for N-dimensional data, such that each bin should contain equal numbers of
        samples. 
        *** Note that due to SciPy's mquantiles() functional design, the equipartion is not strictly true - 
        it operates independently on the marginals, and so with large bin numbers there are usually 
        significant discrepancies from desired behaviour. Fortunately, for TE we find equipartioning is
        extremely beneficial, so we find good accuracy with small bin counts ***
        Args:
            max_bins        -   (int)       The number of bins in each dimension
        Returns:
            bins            -   (dict)      The calculated bin-edges for pdf estimation
                                            using the histogram method, keyed by df column names
        """
        quantiles = np.array([i/max_bins for i in range(0, max_bins+1)])
        bins = dict(zip(self.axes, mquantiles(a=self.df, prob=quantiles, axis=0).T.tolist()))
        
        ## Remove_duplicates
        bins = {k:sorted(set(bins[k])) for (k,v) in bins.items()} 

        if self.lag is not None:
            bins = self.__extend_bins__(bins)
        return bins            
            






# main

lags_ecb = {"Original": 8, "IMF1": 7, "IMF2": 4, "IMF3": 10, "IMF4": 2, "IMF5": 8, "IMF6": 6, "IMF7": 10, "Residual": 10}
#lags_boe = {"Original": 10, "IMF1": 6, "IMF2": 4, "IMF3": 10, "IMF4": 8, "IMF5": 8, "IMF6": 10, "IMF7": 10, "Residual": 10}

total_results = pd.DataFrame()

for bt in vol_est_ecb.columns:
    print(bt)  
    both_ecb = pd.concat([vol_est_ecb[bt], vol_ukx_ecb[bt]], axis = 1)    
    
    signal = both_ecb.iloc[:,0] + (both_ecb.iloc[:,1] * 1j)
    signal.dropna(inplace = True)

    signal = signal.values
    
    decomposer = EmpiricalModeDecomposition(signal, n_imfs = 7)
    imfs = decomposer.decompose()
    imfs_df = pd.DataFrame(imfs)
    imfs_df = imfs_df.T
    
    imfs_set = pd.DataFrame()
    imfs_set["Original_EST"] = both_ecb.iloc[:,0]
    imfs_set["Original_UKX"] = both_ecb.iloc[:,1]
    
    imfs_set.dropna(inplace = True)
    
    for column in range(len(imfs_df.columns)):
        imf_real = np.real(imfs_df.iloc[:,column])
        imf_imag = np.imag(imfs_df.iloc[:,column])
        if column != (len(imfs_df.columns)-1):
            name_real = "IMF"+str(column+1) + "_EST"
            name_imag = "IMF"+str(column+1) + "_UKX"
        else:
            name_real = "Residual_EST"
            name_imag = "Residual_UKX"
        imfs_set[name_real] = imf_real
        imfs_set[name_imag] = imf_imag
    
    time_ts = pd.to_datetime(time_ts, format = '%H:%M:%S')
    indeces = time_ts[imfs_set.index].values
    imfs_set.index = indeces
    
    imfs_set = imfs_set[(imfs_set.index >= datetime.datetime(1900, 1, 1, 12, 45)) &
          (imfs_set.index <= datetime.datetime(1900, 1, 1, 15, 45))]
    
    res = pd.DataFrame()
    
    for col in range(1,len(imfs_set.columns),2):
        x = imfs_set[[imfs_set.columns[col-1], imfs_set.columns[col]]]
        key = imfs_set.columns[col].split("_")[0]
        if key not in lags_ecb.keys():
            getL = 5
        else:
            getL = lags_ecb[key]
        
        # linear
        #TE = TransferEntropy(DF = x, endog = key+"_UKX" , exog = key+"_EST", lag = getL)
        #TE.linear_TE(n_shuffles=200)
        
        #nonlinear
        myAuto = AutoBins(df=x, lag=getL)
        mybins = myAuto.MIC_bins(max_bins=4)
        TE = TransferEntropy(DF = x, endog = key+"_UKX" , exog = key+"_EST", lag = getL)
        TE.nonlinear_TE(pdf_estimator = 'histogram', bins=mybins, n_shuffles=200)
        
        res = pd.concat([res, TE.results.T])
    
    if res.shape[0] < total_results.shape[0]:
        print("skip --- " + str(bt))
        continue
        
    total_results = pd.concat([total_results, res], axis = 1)






     