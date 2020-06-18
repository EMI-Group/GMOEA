import numpy as np


def pm_mutation(pop_dec, boundary):

    pro_m = 1
    dis_m = 20
    pop_dec = pop_dec[:(len(pop_dec)//2)*2, :]
    n, d = np.shape(pop_dec)

    site = np.random.random((n, d)) < pro_m / d
    mu = np.random.random((n, d))
    temp = site & (mu <= 0.5)
    lower, upper = np.tile(boundary[0], (n, 1)), np.tile(boundary[1], (n, 1))

    norm = (pop_dec[temp] - lower[temp]) / (upper[temp] - lower[temp])
    pop_dec[temp] += (upper[temp] - lower[temp]) * \
                           (np.power(2. * mu[temp] + (1. - 2. * mu[temp]) * np.power(1. - norm, dis_m + 1.),
                                     1. / (dis_m + 1)) - 1.)
    temp = site & (mu > 0.5)
    norm = (upper[temp] - pop_dec[temp]) / (upper[temp] - lower[temp])
    pop_dec[temp] += (upper[temp] - lower[temp]) * \
                           (1. - np.power(
                               2. * (1. - mu[temp]) + 2. * (mu[temp] - 0.5) * np.power(1. - norm, dis_m + 1.),
                               1. / (dis_m + 1.)))
    offspring_dec = np.maximum(np.minimum(pop_dec, upper), lower)
    return offspring_dec
