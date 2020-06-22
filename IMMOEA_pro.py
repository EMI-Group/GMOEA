import numpy as np
from Public.uniformweight import uniform_point
from scipy.spatial.distance import cdist


class test_problem(object):
    def __init__(self, m=None, d=None, ref_num=10000):
        self.m = 2
        if d is None:
            self.d = 30
        else:
            self.d = d
        self.lower = np.zeros([1, self.d])
        self.upper = np.ones([1, self.d])
        self.ref_num = ref_num

    def fit(self, operation, in_value):
        if operation == 'init':
            in_value = np.random.random((in_value, self.d)) * \
                      np.tile(self.upper-self.lower, (in_value, 1)) + \
                      np.tile(self.lower, (in_value, 1))
        pop_obj = np.zeros((np.shape(in_value)[0], self.m))

        return in_value, pop_obj

    def pf(self):
        f1 = np.linspace(0, 1, self.ref_num*self.m)
        f2 = 1 - np.sqrt(f1)
        return np.c_[f1, f2]

    def IGD(self, pop_obj):
        return np.mean(np.amin(cdist(self.pf(), pop_obj), axis=1))


class IMMOEA_F1(test_problem):
    def __init__(self, m, d, ref_num):
        test_problem.__init__(self, m, d, ref_num)

    def fit(self, operation='value', in_value=0):
        if operation == 'init':
            in_value = np.random.random([in_value, self.d])
        n, d = np.shape(in_value)
        t = (1 + 5 * np.tile(np.arange(2, d+1), (n, 1))/d) * in_value[:, 1:] - \
            np.tile(in_value[:, 0: 1], (1, d-1))
        g = 1 + 9 * np.mean(t**2, axis=1, keepdims=True)
        pop_obj = np.c_[in_value[:, 0:1],
                        g * (1 - np.sqrt(in_value[:, 0:1] / g))]
        return in_value, pop_obj


class IMMOEA_F2(test_problem):
    def __init__(self, m, d, ref_num):
        test_problem.__init__(self, m, d, ref_num)

    def fit(self, operation='value', in_value=0):
        if operation == 'init':
            in_value = np.random.random([in_value, self.d])
        n, d = np.shape(in_value)
        t = (1 + 5 * np.tile(np.arange(2, d+1), (n, 1))/d) * in_value[:, 1:] - \
            np.tile(in_value[:, 0: 1], (1, d-1))
        g = 1 + 9 * np.mean(t**2, axis=1, keepdims=True)
        pop_obj = np.c_[in_value[:, 0:1],
                        g * (1 - (in_value[:, 0:1] / g)**2)]
        return in_value, pop_obj

    def pf(self):
        f1 = np.linspace(0, 1, self.ref_num*self.m)
        f2 = 1 - f1**2
        return np.c_[f1, f2]


class IMMOEA_F3(test_problem):
    def __init__(self, m, d, ref_num):
        test_problem.__init__(self, m, d, ref_num)

    def fit(self, operation='value', in_value=0):
        if operation == 'init':
            in_value = np.random.random([in_value, self.d])
        n, d = np.shape(in_value)
        t = (1 + 5 * np.tile(np.arange(2, d+1), (n, 1))/d) * in_value[:, 1:] - \
            np.tile(in_value[:, 0: 1], (1, d-1))
        g = 1 + 9 * np.mean(t**2, axis=1, keepdims=True)
        temp = 1 - np.exp(-4*in_value[:, 0: 1]) * np.sin(6*np.pi*in_value[:, 0:1])**6
        pop_obj = np.c_[temp,
                        g * (1 - (temp / g)**2)]
        return in_value, pop_obj

    def pf(self):
        minf1 = np.amin(1 - np.exp(-4*np.linspace(0, 1, 1000000)) *
                        (np.sin(6 * np.pi * np.linspace(0, 1, 1000000)))**6)
        f1 = (np.linspace(minf1, 1, self.ref_num*self.m))
        f2 = 1 - f1**2
        return np.c_[f1, f2]


class IMMOEA_F4(test_problem):
    def __init__(self, m, d, ref_num):
        test_problem.__init__(self, m, d, ref_num)
        self.m = 3

    def fit(self, operation='value', in_value=0):
        if operation == 'init':
            in_value = np.random.random([in_value, self.d])
        n, d = np.shape(in_value)
        t = (1 + 5 * np.tile(np.arange(3, d+1), (n, 1))/d) * in_value[:, 2:] - \
            np.tile(in_value[:, 0: 1], (1, d-2))
        g = np.sum(t**2, axis=1, keepdims=True)
        pop_obj = np.c_[np.cos(np.pi/2*in_value[:, 0: 1])*np.cos(np.pi/2*in_value[:, 1: 2])*(1+g),
                        np.cos(np.pi / 2 * in_value[:, 0: 1]) * np.sin(np.pi / 2 * in_value[:, 1: 2]) * (1 + g),
                        np.sin(np.pi/2*in_value[:, 0: 1])*(1+g)]
        return in_value, pop_obj

    def pf(self):
        f = uniform_point(self.ref_num*self.m, self.m)[0]
        f /= np.tile(np.sqrt(np.sum(f**2, axis=1, keepdims=True)), (1, self.m))
        return f


class IMMOEA_F5(test_problem):
    def __init__(self, m, d, ref_num):
        test_problem.__init__(self, m, d, ref_num)

    def fit(self, operation='value', in_value=0):
        if operation == 'init':
            in_value = np.random.random([in_value, self.d])
        n, d = np.shape(in_value)
        t = in_value[:, 1:]**(1/(1 + 3*np.tile(np.arange(2, d+1), (n, 1))/d)) - \
            np.tile(in_value[:, 0: 1], (1, d-1))
        g = 1 + 9 * np.mean(t**2, axis=1, keepdims=True)
        pop_obj = np.c_[in_value[:, 0: 1],
                        g * (1 - np.sqrt(in_value[:, 0: 1] / g))]

        return in_value, pop_obj


class IMMOEA_F6(test_problem):
    def __init__(self, m, d, ref_num):
        test_problem.__init__(self, m, d, ref_num)

    def fit(self, operation='value', in_value=0):
        if operation == 'init':
            in_value = np.random.random([in_value, self.d])
        n, d = np.shape(in_value)
        t = in_value[:, 1:]**(1/(1+3*np.tile(np.arange(2, d+1), (n, 1))/d)) - \
            np.tile(in_value[:, 0: 1], (1, d-1))
        g = 1 + 9 * np.mean(t**2, axis=1, keepdims=True)
        pop_obj = np.c_[in_value[:, 0: 1],
                        g * (1 -(in_value[:, 0: 1] / g)**2)]

        return in_value, pop_obj


class IMMOEA_F7(IMMOEA_F3):
    def __init__(self, m, d, ref_num):
        IMMOEA_F3.__init__(self, m, d, ref_num)

    def fit(self, operation='value', in_value=0):
        if operation == 'init':
            in_value = np.random.random([in_value, self.d])
        n, d = np.shape(in_value)
        t = in_value[:, 1:]**(1/(1+3*np.tile(np.arange(2, d+1), (n, 1))/d)) - \
            np.tile(in_value[:, 0: 1], (1, d-1))
        g = 1 + 9 * np.mean(t**2, axis=1, keepdims=True)
        temp = 1 - np.exp(-4*in_value[:, 0: 1]) * (np.sin(6*np.pi*in_value[:, 0: 1])**6)
        pop_obj = np.c_[temp,
                        g * (1 - (temp / g)**2)]
        return in_value, pop_obj


class IMMOEA_F8(IMMOEA_F4):
    def __init__(self, m, d, ref_num):
        IMMOEA_F4.__init__(self, m, d, ref_num)

    def fit(self, operation='value', in_value=0):
        if operation == 'init':
            in_value = np.random.random([in_value, self.d])
        n, d = np.shape(in_value)
        t = in_value[:, 2:]**(1/(1+3*np.tile(np.arange(3, d+1), (n, 1))/d)) - \
            np.tile(in_value[:, 0: 1], (1, d-2))
        g = np.sum(t**2, axis=1, keepdims=True)
        pop_obj = np.c_[np.cos(np.pi/2*in_value[:, 0: 1])*np.cos(np.pi/2*in_value[:, 1: 2])*(1+g),
                        np.cos(np.pi / 2 * in_value[:, 0: 1]) * np.sin(np.pi / 2 * in_value[:, 1: 2]) * (1 + g),
                        np.sin(np.pi/2*in_value[:, 0: 1])*(1+g)]
        return in_value, pop_obj


class IMMOEA_F9(test_problem):
    def __init__(self, m, d, ref_num):
        test_problem.__init__(self, m, d, ref_num)
        self.upper = np.c_[1, np.zeros((1, self.d-1)) + 10]

    def fit(self, operation='value', in_value=0):
        if operation == 'init':
            in_value = np.random.random((in_value, self.d)) * \
                       np.tile(self.upper - self.lower, (in_value, 1)) + \
                       np.tile(self.lower, (in_value, 1))
        n, d = np.shape(in_value)
        t = in_value[:, 1:]**(1/(1+3*np.tile(np.arange(2, d+1), (n, 1))/d)) - \
            np.tile(in_value[:, 0: 1], (1, d-1))
        g = np.sum(t**2/4000, axis=1, keepdims=True) - \
            np.prod(np.cos(t/np.tile(np.sqrt(np.arange(1, d)), (n, 1))), axis=1, keepdims=True) + 2
        pop_obj = np.c_[in_value[:, 0: 1],
                        g * (1 - np.sqrt(in_value[:, 0: 1] / g))]

        return in_value, pop_obj


class IMMOEA_F10(IMMOEA_F9):
    def __init__(self, m, d, ref_num):
        IMMOEA_F9.__init__(self, m, d, ref_num)

    def fit(self, operation='value', in_value=0):
        if operation == 'init':
            in_value = np.random.random((in_value, self.d)) * \
                       np.tile(self.upper - self.lower, (in_value, 1)) + \
                       np.tile(self.lower, (in_value, 1))
        n, d = np.shape(in_value)
        t = in_value[:, 1:]**(1/(1+3*np.tile(np.arange(2, d+1), (n, 1))/d)) - \
            np.tile(in_value[:, 0: 1], (1, d-1))
        g = 1 + 10 * (d -1) + np.sum(t ** 2 - 10 * np.cos(2 * np.pi * t), axis=1, keepdims=True)
        pop_obj = np.c_[in_value[:, 0: 1],
                        g * (1 - np.sqrt(in_value[:, 0: 1] / g))]

        return in_value, pop_obj