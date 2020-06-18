import numpy as np


def ea_real(pop_dec, boundary):
    pro_c = 1
    dis_c = 20
    pro_m = 1
    dis_m = 20
    pop_dec = pop_dec[:(len(pop_dec)//2)*2, :]
    (n, d) = np.shape(pop_dec)
    off_n = n//2
    parent_1_dec = pop_dec[:off_n, :]
    parent_2_dec = pop_dec[off_n:, :]
    beta = np.zeros((off_n, d))
    mu = np.random.random((off_n, d))
    beta[mu <= 0.5] = np.power(2*mu[mu <= 0.5], 1/(dis_c + 1))
    beta[mu > 0.5] = np.power(2*mu[mu > 0.5], -1/(dis_c + 1))
    beta = beta * ((-1)**np.random.randint(2, size=(off_n, d)))
    beta[np.random.random((off_n, d)) < 0.5] = 1
    beta[np.tile(np.random.random((off_n, 1)) > pro_c, (1, d))] = 1
    pop_dec = np.vstack(((parent_1_dec + parent_2_dec) / 2 + beta * (parent_1_dec - parent_2_dec) / 2,
                               (parent_1_dec + parent_2_dec) / 2 - beta * (parent_1_dec - parent_2_dec) / 2))

    offspring_dec = pop_dec
    site = np.random.random((n, d)) < pro_m / d
    mu = np.random.random((n, d))
    temp = site & (mu <= 0.5)
    lower, upper = np.tile(boundary[0], (n, 1)), np.tile(boundary[1], (n, 1))

    norm = (offspring_dec[temp] - lower[temp]) / (upper[temp] - lower[temp])
    offspring_dec[temp] += (upper[temp] - lower[temp]) * \
                           (np.power(2. * mu[temp] + (1. - 2. * mu[temp]) * np.power(1. - norm, dis_m + 1.),
                                     1. / (dis_m + 1)) - 1.)
    temp = site & (mu > 0.5)
    norm = (upper[temp] - offspring_dec[temp]) / (upper[temp] - lower[temp])
    offspring_dec[temp] += (upper[temp] - lower[temp]) * \
                           (1. - np.power(
                               2. * (1. - mu[temp]) + 2. * (mu[temp] - 0.5) * np.power(1. - norm, dis_m + 1.),
                               1. / (dis_m + 1.)))
    offspring_dec = np.maximum(np.minimum(offspring_dec, upper), lower)
    return offspring_dec
