import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation

from pymoo.factory import get_termination
import matplotlib.pyplot as plt
from pymoo.optimize import minimize

from src.models.d_search.utils.utestfun import TestFuncs

tf = TestFuncs()
testfun = tf.get('t3')

class MyProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=testfun['n_var'],
                         n_obj=testfun['n_var'],
                         n_constr=0,
                         xl=np.array(testfun['lb']),
                         xu=np.array(testfun['ub']))

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = testfun['f'](x)


problem = MyProblem()

if __name__ == '__main__':
    #%%
    algorithm = NSGA2(
        pop_size=100,
        n_offsprings=100,
        sampling=get_sampling("real_random"),
        crossover=get_crossover("real_sbx", prob=0.9, eta=15),
        mutation=get_mutation("real_pm", eta=20),
        eliminate_duplicates=True
    )

    termination = get_termination("n_gen", 100)

    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1,
                   save_history=True,
                   verbose=True)

    X = res.X
    F = res.F

    # %%
    xl, xu = problem.bounds()
    plt.figure(figsize=(7, 5))
    plt.scatter(X[:, 0], X[:, 1], s=30, facecolors='none', edgecolors='r')
    plt.xlim(xl[0], xu[0])
    plt.ylim(xl[1], xu[1])
    plt.title("Design Space")
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
    plt.title("Objective Space")
    plt.show()
