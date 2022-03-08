from src.models.d_search.utils.oper import squared_norm_mul_along_axis


class Termination:
    def __init__(self, name):
        self.name = name

    def has_next(self, algorithm):
        raise NotImplementedError


class MaxIter(Termination):

    def __init__(self, maxiter):
        self.maxiter = maxiter

        super().__init__('n_iter')

    def has_next(self, algorithm):
        if algorithm.iter >= self.maxiter:
            print('iterations exceeded: ended with {} it'.format(algorithm.iter))
            return False
        return True


class Tol(Termination):
    def __init__(self, tol):
        self.tol = tol

        super().__init__('tol')

    def has_next(self, algorithm):
        tol = squared_norm_mul_along_axis(algorithm.a, algorithm.dx)
        if tol >= self.tol and algorithm.iter > 1:
            print('tol exceeded: ended with {} it'.format(algorithm.iter))
            return False
        return True


class NullTermination(Termination):
    def __init__(self):
        super().__init__(name='None')
