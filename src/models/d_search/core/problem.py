import numpy as np
from pymoo.core.problem import ElementwiseProblem, Problem


class ContinuationProblem(ElementwiseProblem):

    def __init__(self, f, df, n_var, n_obj, xl, xu):
        self.f = f
        self.df = df
        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_constr=0,
                         xl=np.array(xl),
                         xu=np.array(xu))

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.f(x)

    def gradient(self, x):
        return self.df(x)


if __name__ == '__main__':
    # %%
    from src.models.d_search.utils.utestfun import TestFuncs

    tf = TestFuncs()
    testfun = tf.get('t3')
    problem = ContinuationProblem(f=testfun['f'],
                                  df=testfun['df'],
                                  n_var=testfun['n_var'],
                                  n_obj=testfun['n_var'],
                                  xl=testfun['lb'],
                                  xu=testfun['ub'])

    res = {}
    X = np.random.random((10, 2))
    new_res = problem.do(X, res)
