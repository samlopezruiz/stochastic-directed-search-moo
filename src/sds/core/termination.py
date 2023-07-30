from src.sds.utils.functions import squared_norm_mul_along_axis


class Termination:
    def __init__(self, name):
        self.name = name

    def has_next(self, algorithm):
        raise NotImplementedError


class MaxIter(Termination):

    def __init__(self, maxiter):
        self.maxiter = maxiter

        super().__init__('iterations')

    def has_next(self, algorithm):
        return algorithm.iter < self.maxiter and not algorithm.end_flag


class Tol(Termination):
    def __init__(self, tol):
        self.tol = tol

        super().__init__('tol')

    def has_next(self, algorithm):
        tol = squared_norm_mul_along_axis(algorithm.a, algorithm.dx)
        return not (tol >= self.tol and algorithm.iter > 1) and not algorithm.end_flag


class NullTermination(Termination):
    def __init__(self, *args, **kwargs):
        super().__init__(name='None')

    def has_next(self, algorithm):
        return not algorithm.end_flag
