import numpy as np
import pandas as pd
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
import seaborn as sns
from pymoo.factory import get_termination
import matplotlib.pyplot as plt
from pymoo.optimize import minimize

from src.models.d_search.algorithms.continuation import DsContinuation
from src.models.d_search.core.problem import ContinuationProblem
from src.models.d_search.utils.factory import get_tfun, get_corrector, get_predictor
from src.models.d_search.utils.utestfun import TestFuncs

if __name__ == '__main__':
    tf = TestFuncs()
    testfun = tf.get('t3')
    problem = ContinuationProblem(f=testfun['f'],
                                  df=testfun['df'],
                                  n_var=testfun['n_var'],
                                  n_obj=testfun['n_var'],
                                  xl=testfun['lb'],
                                  xu=testfun['ub'])

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

    ixs = np.argsort(F[:, 0])
    F_sorted = F[ixs]
    X_sorted = X[ixs]

    # %%
    # x0 = [0.01741, -0.24136]
    x0 = [-1, -1]
    f0 = problem.evaluate(x0)

    # xl, xu = problem.bounds()
    # plt.figure(figsize=(7, 5))
    # plt.scatter(X[:, 0], X[:, 1], s=30, facecolors='none', edgecolors='r')
    # plt.scatter(x0[0], x0[1], s=30, facecolors='b', edgecolors='b')
    # plt.xlim(xl[0], xu[0])
    # plt.ylim(xl[1], xu[1])
    # plt.title("Design Space")
    # plt.show()
    #
    # plt.figure(figsize=(7, 5))
    # plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
    # plt.scatter(f0[0, 0], f0[0, 1], s=30, facecolors='red', edgecolors='red')
    # plt.title("Objective Space")
    # plt.show()

    # %%
    corrector = get_corrector('ds',
                              problem=problem,
                              t_fun=get_tfun('szc5', problem=problem),
                              a_fun=lambda a, dx: a,
                              maxiter=20)

    predictor = get_predictor('left',
                              problem=problem)

    ds_cont = DsContinuation(problem=problem,
                             predictor=predictor,
                             corrector=corrector,
                             dir=-1,
                             eps=10,
                             termination=('n_iter', 1000),
                             history=True,
                             )

    res = ds_cont.run(x0)

    #%%
    markersize = 40
    # fig, ax = plt.subplots()
    # plt.scatter(res['X_p'][:, 0], res['X_p'][:, 1], s=markersize, facecolors='none', edgecolors='b', label='predictor')
    # plt.scatter(res['X_c'][:, 0], res['X_c'][:, 1], s=markersize, facecolors='none', edgecolors='r', label='corrector')
    # plt.title('Decision space')
    # plt.legend()
    # plt.show()
    #
    # fig, ax = plt.subplots()
    # plt.scatter(res['F_p'][:, 0], res['F_p'][:, 1], s=markersize, facecolors='none', edgecolors='b', label='predictor')
    # plt.scatter(res['F_c'][:, 0], res['F_c'][:, 1], s=markersize, facecolors='none', edgecolors='r', label='corrector')
    # plt.title('Objective space')
    # plt.legend()
    # plt.show()
    #%%
    # data = pd.DataFrame(res['X'])
    # sns.scatterplot(data=data, x=0, y=1, hue=data.index)
    # plt.title('Decision space')
    # plt.show()
    #
    # data = pd.DataFrame(res['F'])
    # sns.scatterplot(data=data, x=0, y=1, hue=data.index)
    # plt.title('Objective space')
    # plt.show()

    #%%
    data1 = pd.DataFrame(res['X'])
    data1['method'] = 'cont'

    data2 = pd.DataFrame(X_sorted)
    data2['method'] = 'moea'

    data = pd.concat([data2, data1], axis=0)
    sns.scatterplot(data=data, x=0, y=1, hue='method')
    plt.title('Decision space')
    plt.show()

    data1 = pd.DataFrame(res['F'])
    data1['method'] = 'cont'

    data2 = pd.DataFrame(F_sorted)
    data2['method'] = 'moea'

    data = pd.concat([data2, data1], axis=0)
    sns.scatterplot(data=data, x=0, y=1, hue='method')
    plt.title('Objective space')
    plt.show()

    # F_sorted
    # markersize = 40
    # fig, ax = plt.subplots()
    # plt.scatter(res['X'][:, 0], res['X'][:, 1], s=markersize, facecolors='none', edgecolors='b')
    # plt.title('Decision space')
    # plt.show()
    #
    # fig, ax = plt.subplots()
    # plt.scatter(res['F'][:, 0], res['F'][:, 1], s=markersize, facecolors='none', edgecolors='b')
    # plt.title('Objective space')
    # plt.show()