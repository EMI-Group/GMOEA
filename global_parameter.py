class GlobalParameter:
    """
    This class includes the general parameters for running the algorithm.
    We can also define the class of population, which includes all the operations
    """
    def __init__(self, m=2, n=100, d=3, eva=10000, decs=None, operator=None, pro=None, run=None):
        self.m = m
        self.n = n
        self.pro = pro(d=d, m=m)     # Initialize the class of problem
        self.d = self.pro.d                         # objectives
        self.lower = self.pro.lower
        self.upper = self.pro.upper
        self.boundary = (self.lower, self.upper)
        self.eva = eva
        self.operator = operator
        self.result = decs
        self.decs = []
        self.run = run
