import os
import time

import numpy as np
import plotly.io as pio
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.factory import get_termination
from pymoo.optimize import minimize

from src.models.d_search.algorithms.continuation import DsContinuation
from src.models.d_search.core.problem import ScalarContinuationProblem
from src.models.d_search.utils.plot import plot_2D_points_vectors, plot_2D_predictor_corrector
from src.models.d_search.utils.factory import get_tfun, get_corrector, get_predictor
from src.models.d_search.utils.utestfun import TestFuncs

pio.renderers.default = "browser"
if __name__ == '__main__':
    # %%
    cfg = {'save_plots': False,
           'problem_name': 't5a1'
           }

    solve_moea = False

    tfun = TestFuncs()
    testfun = tfun.get(cfg['problem_name'])
    problem = ScalarContinuationProblem(f=testfun['f'],
                                        df=testfun['df'],
                                        n_var=testfun['n_var'],
                                        n_obj=testfun['n_obj'],
                                        xl=testfun['lb'],
                                        xu=testfun['ub'])

    # if 'pf' in testfun and isinstance(testfun)
    pf = testfun.get('pf', None)
    ps = testfun.get('ps', None)
    # ps = testfun['ps']
    # pf = pd.read_csv(os.path.join('ref', testfun['pf']), header=None) if 'pf' in testfun else None
    # ps = pd.read_csv(os.path.join('ref', testfun['ps']), header=None) if 'ps' in testfun else None

    # %%
    if solve_moea:
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
        pf = F[ixs]
        ps = X[ixs]

    # fig, ax = plt.subplots()
    # ax.plot(pf[:, 0], pf[:, 1], 'o')
    # plt.title('objective space')
    # plt.show()
    #
    # fig, ax = plt.subplots()
    # ax.plot(ps[:, 0], ps[:, 1], 'o')
    # plt.title('decision space')
    # plt.show()

    # %%
    x0 = [-0.5, -0.5]
    # x0 = [1., 1.]
    f0 = problem.evaluate(x0)

    predictor = get_predictor('no_adjustment',
                              problem=problem,
                              eps=1e-2)

    corrector = get_corrector('delta_criteria',
                              problem=problem,
                              t_fun=get_tfun('test',
                                             problem=problem,
                                             eps=1e-5,
                                             maxiter=50),
                              a_fun=lambda a, dx: a,
                              eps=1e-1,
                              maxiter=100)

    ds_cont = DsContinuation(problem=problem,
                             predictor=predictor,
                             corrector=corrector,
                             tol_eps=1e-3,
                             dir=-1,
                             step_eps=1e-1,
                             termination=('n_iter', 100),
                             debug=True,
                             history=True,
                             )

    t0 = time.time()
    res = ds_cont.run(x0)
    print('time: {} s'.format(round(time.time() - t0, 4)))
    print('f(x) evals: {}, dx(x) evals: {}'.format(problem.n_f_evals, problem.n_grad_evals))

    # %%
    file_path = os.path.join('../img')
    plot_2D_predictor_corrector(points=res['X'],
                                predictors=res['X_p'],
                                correctors=res['X_c'],
                                pareto=ps,
                                arrow_scale=0.4,
                                point_name='x',
                                markersize=6,
                                line_width=1.5,
                                plot_arrows=True,
                                plot_points=True,
                                save=cfg['save_plots'],
                                file_path=os.path.join('../img',
                                                       'decision_space_{}'.format(cfg['problem_name'])),
                                )

    plot_2D_predictor_corrector(points=res['F'],
                                predictors=res['F_p'],
                                correctors=res['F_c'],
                                pareto=pf,
                                arrow_scale=0.4,
                                point_name='f(x)',
                                markersize=6,
                                line_width=1.5,
                                plot_arrows=True,
                                plot_points=True,
                                save=cfg['save_plots'],
                                file_path=os.path.join('../img',
                                                       'objective_space_{}'.format(cfg['problem_name'])),
                                )

    # %%
    plot_2D_points_vectors(res['X_r' if 'X_r' in res else 'X'],
                           vectors=res['vs'],
                           pareto=ps,
                           scale=0.05,
                           arrow_scale=0.4,
                           point_name='x',
                           vector_name='v')

    plot_2D_points_vectors(res['F_r' if 'F_r' in res else 'F'],
                           vectors=res['as'],
                           pareto=pf,
                           scale=0.5,
                           arrow_scale=0.4,
                           point_name='f(x)',
                           vector_name='a')
