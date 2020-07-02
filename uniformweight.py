import numpy as np
from scipy.special import comb
from itertools import combinations as n_choose_k


def uniform_weight(h1, h2, m):
    """
   :param m: number of objectives
   :param h1: control parameter
   :param h2: control parameter
   :return: uniform weight vectors
   """
    w, n = weight(h1, m)
    if h2 > 0:
        w2, n2 = weight(h2, m)
        n = n + n2
        w = np.r_[w, w2 / 2.0 + 1.0 / (2.0 * m)]

    w = np.maximum(w, 1e-6)
    return w, n


def weight(h, m):
    n = comb(h+m-1, m-1).astype(int)
    temp = np.array(list(n_choose_k(range(1, h+m), m-1))) - \
           np.tile(np.arange(m-1), (comb(h+m-1, m-1).astype(int), 1)) - 1
    w = np.zeros([n, m])
    w[:, 0] = temp[:, 0] - 0
    for i in range(1, m-1):
        w[:, i] = temp[:, i] - temp[:, i-1]
    w[:, -1] = h - temp[:, -1]
    w /= h
    return w, n


def uniform_point(n, m):
    h1 = 1
    while comb(h1 + m, m - 1) <= n:
        h1 += 1
    w = np.array(list(n_choose_k(range(1, h1 + m), m-1))) - \
        np.tile(np.array(range(m-1)), (comb(h1+m-1, m-1).astype(int), 1)) - 1
    w = (np.c_[w, np.zeros((np.shape(w)[0], 1)) + h1] -
         np.c_[np.zeros((np.shape(w)[0], 1)), w]) / h1
    if h1 < m:
        h2 = 0
        while comb(h1+m-1, m-1) + comb(h2+m, m-1) <= n:
            h2 += 1
        if h2 > 0:
            w2 = np.array(list(n_choose_k(range(1, h2+m), m-1))) - \
                 np.tile(np.array(range(m - 1)), (comb(h2+m-1, m-1), 1)) - 1
            w2 = (np.c_[w2, np.zeros((np.shape(w2)[0], 1)) + h2] -
                  np.c_[np.zeros((np.shape(w2)[0], 1)), w2]) / h2
            w = np.r_[w, w2/2. + 1./(2.*m)]
    w = np.maximum(w, 1e-6)
    n = np.shape(w)[0]
    return w, n
