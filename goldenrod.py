import numpy as np


def get_msd(x):
    """Compute the mean squared displacement.
    """
    msd = []
    for i in range(1, len(x)):
        msd.append(np.average((x[i:] - x[:-i])**2))
    return np.array(msd)


def get_2d_msd(x, y):
    return get_msd(x) + get_msd(y)


def get_xycols(df, subtract_init=False):
    """Extract the x and y columns from xls file, assuming usual headers.
    """
    x = np.array(df['centroid_1 (x) nm'])
    y = np.array(df['centroid_2 (y) (nm)'])
    if subtract_init:
        x -= x[0]
        y -= y[0]
    return x, y


def get_tcol(df):
    return np.array(df['real time (s)'])


def pvariation(x, p, n, t=None):
    """p-variation for finite n.

    Frames beyond the highest power of 2 available are ignored.
    All intervals are constructed to be of equal length. T is assumed,
    without loss of generality, to equal the total number of frames included.

    """
    maxn = int(np.log2(len(x)) // 1)
    T = 2**maxn
    if t is None:
        t = T
    dt = T // 2**n
    ts = np.arange(2**n) * dt
    ts = ts[ts < t]  # Ignore frames greater than t
    total = 0.
    for i in range(len(ts) - 1):
        total += np.abs(x[ts[i + 1]] - x[ts[i]])**p
    return total
