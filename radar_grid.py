import numpy as np


def is_member_rows(a, b):
    """
    Equivalent of 'ismember' from Matlab
    a.shape = (nRows_a, nCol)
    b.shape = (nRows_b, nCol)
    return the idx where b[idx] == a
    """
    return np.nonzero(np.all(b == a[:, np.newaxis], axis=2))[1]


def radar_grid(pop_obj, div):
    """
    the RadViz based projection and division
    :param pop_obj: the input objective vector
    :param div: the number of divisions for each axis
    :return: the projected location and grid label, site starts from 0
    """
    n, m = np.shape(pop_obj)
    theta = np.arange(0.0, 2.*np.pi, 2.*np.pi/m)
    r_loc = np.zeros((n, 2))
    r_loc[:, 0] = np.sum(pop_obj * np.tile(np.cos(theta), (n, 1)), axis=1) / np.sum(pop_obj, axis=1)
    r_loc[:, 1] = np.sum(pop_obj * np.tile(np.sin(theta), (n, 1)), axis=1) / np.sum(pop_obj, axis=1)
    r_loc = (r_loc + 1)/2
    y_lower = np.amin(r_loc, axis=0)
    y_upper = np.amax(r_loc, axis=0)
    if any(y_lower == y_upper):
        norm_r = r_loc
    else:
        norm_r = (r_loc - np.tile(y_lower, (n, 1)))/np.tile(y_upper-y_lower, (n, 1))

    g_loc = np.floor(norm_r * div)
    g_loc[g_loc >= div] = div - 1
    uni_g, _ = np.unique(g_loc[:, 0], return_inverse=True)
    a = np.unique(g_loc, axis=0)
    uni_g = a[a[:, 0].argsort(), ]
    site = is_member_rows(g_loc, uni_g)
    return site, r_loc
