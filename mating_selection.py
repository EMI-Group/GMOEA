import numpy as np
from Public.tournament import tournament
from Public.radar_grid import radar_grid
from scipy.stats import itemfreq

def mating_selection(population, Range, n):
    """
    Mating selection in RSEA
    :param population: current population
    :param n: number of selected individuals
    :param Range: the range of the objective vectors
    :return: next generation population
    """
    pop_obj = population[1]
    N = np.shape(pop_obj)[0]
    pop_obj = (pop_obj - np.tile(Range[0], (N, 1))) / \
              np.tile(Range[1] - Range[0], (N, 1))
    con = np.sqrt(np.sum(pop_obj**2, axis=1))
    site, _ = radar_grid(pop_obj, np.ceil(np.sqrt(N)))
    crowd_g = itemfreq(site)[:, 1]

    mating_pool = np.zeros(np.ceil(N/2).astype(int)*2)
    grids = tournament(2, len(mating_pool), crowd_g.reshape((crowd_g.size, 1)))
    for i in range(len(mating_pool)):
        current = np.nonzero(site == grids[i])[0]
        if current is None:
            mating_pool[i] = np.random.randint(0, N, 1)
        else:
            parents = current[np.random.randint(0, len(current), 4)]
            best = np.argmin(con[parents])
            mating_pool[i] = parents[best]
    return mating_pool.astype(int)


