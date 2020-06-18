import numpy as np
from scipy.spatial.distance import cdist


def environment_selection(population, n):
    """
    environmental selection in SPEA-2
    :param population: current population
    :param n: number of selected individuals
    :return: next generation population
    """
    fitness = cal_fit(population[1])
    index = np.nonzero(fitness < 1)[0]
    if len(index) < n:
        rank = np.argsort(fitness)
        index = rank[: n]
    elif len(index) > n:
        del_no = trunc(population[1][index, :], len(index) - n)
        index = np.setdiff1d(index, index[del_no])

    population = [population[0][index, :], population[1][index, :]]
    return population, index


def trunc(pop_obj, k):
    n, m = np.shape(pop_obj)
    distance = cdist(pop_obj, pop_obj)
    distance[np.eye(n) > 0] = np.inf
    del_no = np.ones(n) < 0
    while np.sum(del_no) < k:
        remain = np.nonzero(np.logical_not(del_no))[0]
        temp = np.sort(distance[remain, :][:, remain], axis=1)
        rank = np.argsort(temp[:, 0])
        del_no[remain[rank[0]]] = True
    return del_no


def cal_fit(pop_obj):
    n, m = np.shape(pop_obj)
    dominance = np.ones((n, n)) < 0
    for i in range(0, n-1):
        for j in range(i+1, n):
            k = int(np.any(pop_obj[i, :] < pop_obj[j, :])) - int(np.any(pop_obj[i, :] > pop_obj[j, :]))
            if k == 1:
                dominance[i, j] = True
            elif k == -1:
                dominance[j, i] = True

    s = np.sum(dominance, axis=1, keepdims=True)

    r = np.zeros(n)
    for i in range(n):
        r[i] = np.sum(s[dominance[:, i]])

    distance = cdist(pop_obj, pop_obj)
    distance[np.eye(n) > 0] = np.inf
    distance = np.sort(dominance, axis=1)
    d = 1 / (distance[:, int(np.sqrt(n))] + 2)

    fitness = r + d
    return fitness

