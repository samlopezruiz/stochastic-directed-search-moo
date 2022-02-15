import numpy as np

funcs = {}

funcs['t3'] = {'f': lambda X: [(X[0] - 1) ** 4 + (X[1] - 1) ** 2, (X[0] + 1) ** 2 + (X[1] + 1) ** 2],
               'df': lambda X: np.array([[4 * (X[0] - 1) ** 3, 2 * (X[1] - 1)],
                                         [2 * (X[0] + 1), 2 * (X[1] + 1)]]),
               'lb': [-5, -5],
               'ub': [5, 5],
               'n_var': 2,
               'n_obj': 2
               }


class TestFuncs:

    def __init__(self):
        self.funcs = funcs

    def get(self, key):
        return self.funcs[key]


if __name__ == '__main__':
    tf = TestFuncs()
    testfun = tf.get('t3')
    df = testfun['df']
    x = [1, -1]
    dx = df(x)
