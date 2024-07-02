"""Code for generating and analyzing anomalous diffusion trajectories.

Adapted from https://github.com/AnomDiffDB/DB
Granik, N., Weiss, L.E., Nehme, E., Levin, M., Chein, M., Perlson, E.,
Roichman, Y. and Shechtman, Y., 2019. Single particle diffusion
characterization by deep learning. Biophysical Journal.
"""

import numpy as np
from scipy import stats, fftpack
# import stochastic
from scipy.optimize import curve_fit
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.image as mpimg
from scipy.stats import levy_stable
# import pickle
# import pandas as pd
from scipy.stats import ttest_ind_from_stats
from scipy.stats import pearsonr


def OrnsteinUng(n=1000, T=50, speed=0, mean=0, vol=0):
    """
    Function OrnsteinUng generates a single realization of the
    Ornstein–Uhlenbeck noise process
    see https://stochastic.readthedocs.io/en/latest/diffusion.html
    #stochastic.diffusion.OrnsteinUhlenbeckProcess
    for more details.

    Note that the stochastic variable starts at the mean value.

    Input:
        n - number of points to generate
        T - End time
        speed - speed of reversion
        mean - mean of the process
        vol - volatility coefficient of the process

    Outputs:
        x - Ornstein Uhlenbeck process realization
    """
    OU = diffusion.OrnsteinUhlenbeckProcess(
        speed=speed, mean=mean, vol=vol, t=T)
    x = OU.sample(n=n, initial=mean)
    return x


def MSD(x, y):
    """
    Input:
        x,y - x,y positions of a localized particle over time
        (assuming constant dt between frames)

    Outputs:
        tVec - vector of time points
        msd - mean square displacement curve
        a - anomalous exponent (=2*Hurst exponent for FBM)
    """
    data = np.sqrt(x**2 + y**2)
    nData = np.size(data)
    numberOfDeltaT = np.int((nData - 1))
    tVec = np.arange(1, np.int(numberOfDeltaT * 0.9))

    msd = np.zeros([len(tVec), 1])
    for dt, ind in zip(tVec, range(len(tVec))):
        sqdisp = (data[1 + dt:] - data[:-1 - dt])**2
        msd[ind] = np.mean(sqdisp, axis=0)

    msd = np.array(msd)
    a, b = curve_fit(curve_func, np.log(tVec), np.log(msd.ravel()))
    return tVec, msd, a[1]


def Sub_brownian(x0, n, dt, delta, out=None):
    x0 = np.asarray(x0)
    # generate a sample of n numbers from a normal distribution.
    # to generate an instance of Brownian motion (i.e. the Wiener process):

    #    X(t) = X(0) + r(0, delta**2 * t; 0, t)

    # where r(a,b; t0, t1) is a normally distributed random variable
    # with mean a and variance b.
    # X(t), has a normal distribution whose mean is
    # the position at time t=0 and whose variance is delta**2*t.
    r = stats.norm.rvs(size=x0.shape + (n,), scale=delta * np.sqrt(dt))
    # https://scipy-cookbook.readthedocs.io/items/BrownianMotion.html#2D-Brownian-Motion
    # If `out` was not given, create an output array.
    if out is None:
        out = np.empty(r.shape)
    # Compute Brownian motion by forming the cumulative sum of random samples.
    np.cumsum(r, axis=-1, out=out)
    # Add the initial condition.
    out += np.expand_dims(x0, axis=-1)

    return out


def Brownian(N=1000, T=50, delta=0.01):
    '''
    Brownian - generate Brownian motion trajectory (x,y)

    Inputs:
        N - number of points to generate
        T - End time
        delta - Diffusion coefficient

    Outputs:
        out1 - x axis values for each point of the trajectory
        out2 - y axis values for each point of the trajectory
    '''
    x = np.empty((2, N + 1))
    x[:, 0] = 0.0

    Sub_brownian(x[:, 0], N, T / N, delta, out=x[:, 1:])

    out1 = x[0]
    out2 = x[1]

    return out1, out2


def fbm_diffusion(n=1000, H=1, T=15):
    '''
    function fbm_diffusion generates FBM diffusion trajectory (x,y,t)
    realization is based on the Circulant Embedding method presented in:
    Schmidt, V. 2014. Stochastic geometry, spatial statistics and random fields

    Input:
        n - number of points to generate
        H - Hurst exponent
        T - end time

    Outputs:
        x - x axis coordinates
        y - y axis coordinates
        t - time points

    '''

    # first row of circulant matrix
    r = np.zeros(n + 1)
    r[0] = 1
    idx = np.arange(1, n + 1, 1)
    r[idx] = 0.5 * ((idx + 1)**(2 * H) - 2 * idx**(2 * H) + (idx - 1)**(2 * H))
    r = np.concatenate((r, r[np.arange(len(r) - 2, 0, -1)]))

    # get eigenvalues through fourier transform
    lamda = np.real(fftpack.fft(r)) / (2 * n)
    lamda[lamda < 0] = 0

    # get trajectory using fft: dimensions assumed uncoupled
    x = fftpack.fft(np.sqrt(lamda) * (np.random.normal(
        size=(2 * n)) + 1j * np.random.normal(size=(2 * n))))
    x = n**(-H) * np.cumsum(np.real(x[:n]))  # rescale
    x = ((T**H) * x)  # resulting traj. in x
    y = fftpack.fft(np.sqrt(lamda) * (np.random.normal(
        size=(2 * n)) + 1j * np.random.normal(size=(2 * n))))
    y = n**(-H) * np.cumsum(np.real(y[:n]))  # rescale
    y = ((T**H) * y)  # resulting traj. in y

    t = np.arange(0, n + 1, 1) / n
    t = t * T  # scale for final time T

    return x, y, t, lamda


# Generate mittag-leffler random numbers
def mittag_leffler_rand(beta=0.5, n=1000, gamma=1):
    t = -np.log(np.random.uniform(size=[n, 1]))
    u = np.random.uniform(size=[n, 1])
    w = np.sin(beta * np.pi) / np.tan(beta * np.pi * u) - np.cos(beta * np.pi)
    # t = t * ((w**1 / (beta)))
    t = t * w**(1. / beta)
    t = gamma * t

    return t


# Generate symmetric alpha-levi random numbers
def symmetric_alpha_levy(alpha=0.5, n=1000, gamma=1):
    u = np.random.uniform(size=[n, 1])
    v = np.random.uniform(size=[n, 1])

    phi = np.pi * (v - 0.5)
    w = np.sin(alpha * phi) / np.cos(phi)
    z = -1 * np.log(u) * np.cos(phi)
    z = z / np.cos((1 - alpha) * phi)
    x = gamma * w * z**(1 - (1 / alpha))

    return x


# needed for CTRW
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


# Generate CTRW diffusion trajectory
def CTRW(n=1000, alpha=1, gamma=1, T=40):
    '''
    CTRW diffusion - generate CTRW trajectory (x,y,t)
    function based on mittag-leffler distribution for waiting times and
    alpha-levy distribution for spatial lengths.
    for more information see:
    Fulger, D., Scalas, E. and Germano, G., 2008.
    Monte Carlo simulation of uncoupled continuous-time random walks yielding a
    stochastic solution of the space-time fractional diffusion equation.
    Physical Review E, 77(2), p.021122.

    https://en.wikipedia.org/wiki/Lévy_distribution
    https://en.wikipedia.org/wiki/Mittag-Leffler_distribution

    Inputs:
        n - number of points to generate
        alpha - exponent of the waiting time distribution function
        gamma  - scale parameter for the mittag-leffler and alpha stable
                 distributions.
        T - End time
    '''
    jumpsX = mittag_leffler_rand(alpha, n, gamma)

    rawTimeX = np.cumsum(jumpsX)
    tX = rawTimeX * (T) / np.max(rawTimeX)
    tX = np.reshape(tX, [len(tX), 1])

    x = symmetric_alpha_levy(alpha=2, n=n, gamma=gamma**(alpha / 2))
    x = np.cumsum(x)
    x = np.reshape(x, [len(x), 1])

    y = symmetric_alpha_levy(alpha=2, n=n, gamma=gamma**(alpha / 2))
    y = np.cumsum(y)
    y = np.reshape(y, [len(y), 1])

    tOut = np.arange(0, n, 1) * T / n
    xOut = np.zeros([n, 1])
    yOut = np.zeros([n, 1])
    for i in range(n):
        xOut[i, 0] = x[find_nearest(tX, tOut[i]), 0]
        yOut[i, 0] = y[find_nearest(tX, tOut[i]), 0]
    return xOut.T[0], yOut.T[0], tOut


