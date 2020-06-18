import numpy as np


def nd_sort(pop_obj, n_sort=1):
    """

    :param pop_obj: the objective vectors
    :param n_sort: sort n_sort solutions of existing vectors
    :return: the maximum front number and the front ranking
    this function is cpu only as some subfunctions are supported by cpu
    """

    _, b, loc = np.unique(pop_obj[:, 0], return_index=True, return_inverse=True)
    pop_obj = pop_obj[b, :]
    table = np.histogram(loc, bins=range(np.max(loc)+2))[0]
    n, m_obj = np.shape(pop_obj)
    rank = np.arange(n)
    front_no = np.inf * np.ones(n)
    max_front = 0
    while np.sum(table[front_no < np.inf]) < min(n_sort, len(loc)):
        max_front += 1
        for i in range(n):
            if front_no[i] == np.inf:
                dominated = False
                for j in range(i, 0, -1):
                    if front_no[j - 1] == max_front:
                        m = 2
                        while (m <= m_obj) and (pop_obj[i, m - 1] >= pop_obj[j - 1, m - 1]):
                            m += 1
                        dominated = m > m_obj
                        if dominated or (m_obj == 2):
                            break
                if not dominated:
                    front_no[i] = max_front
        front_no[rank] = front_no
    return front_no[loc], max_front
