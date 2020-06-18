import numpy as np


def tournament(k_size, n_size, fit):
    """
    tournament selection
    :param k_size: number of solutions to be compared
    :param n_size: number of solutions to be selected
    :param fit: fitness vectors, the smaller the better
    :return: index of selected solutions
    """
    n, m = np.shape(fit)
    mate = np.zeros(n_size, dtype=int)
    for i in range(n_size):
        a = np.random.randint(n)
        for j in range(k_size):
            b = np.random.randint(n)
            for r in range(m):
                if fit[b, r] < fit[a, r]:
                    a = b
        mate[i] = a

    return mate
