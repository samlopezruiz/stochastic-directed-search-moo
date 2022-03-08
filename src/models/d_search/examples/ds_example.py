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
from src.models.d_search.utils.plot import plot_2D_points_vectors, plot_2D_predictor_corrector, plot_2D_points_traces
from src.models.d_search.utils.factory import get_tfun, get_corrector, get_predictor, get_cont_termination
from src.models.d_search.utils.utestfun import TestFuncs

pio.renderers.default = "browser"
if __name__ == '__main__':
    # %%
    cfg = {'save_plots': False,
           'problem_name': 't3_ag'
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
        pf_moea = F[ixs]
        ps_moea = X[ixs]

    # %%
    # x0 = [-0.10360, -0.48965]
    x0 = [1., 1.]
    # x0 = [-1., -1.]

    predictor = get_predictor('no_adjustment', problem=problem)

    corrector = get_corrector('delta_criteria',
                              problem=problem,
                              t_fun=get_tfun('test',
                                             problem=problem,
                                             eps=1e-5,
                                             maxiter=50),
                              a_fun=lambda a, dx: a,
                              eps=1e-3,
                              maxiter=10)

    ds_cont = DsContinuation(problem=problem,
                             predictor=predictor,
                             corrector=corrector,
                             dir=1,
                             step_eps=5e-1,
                             termination=get_cont_termination('tol', tol=5e-3),
                             debug=True,
                             history=True,
                             )

    t0 = time.time()
    problem.n_f_evals, problem.n_grad_evals = 0, 0
    results = ds_cont.run(x0)
    print('time: {} s'.format(round(time.time() - t0, 4)))
    print('f(x) evals: {}, dx(x) evals: {}'.format(problem.n_f_evals, problem.n_grad_evals))
    print(results['evaluations'])

    # %%
    file_path = os.path.join('../img')
    res = results['population']
    plot_2D_predictor_corrector(points=res['X'],
                                predictors=res['X_p'],
                                correctors=res['X_c'],
                                pareto=testfun['ps'],
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
                                pareto=testfun['pf'],
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
    plot_2D_points_vectors(res['X'],
                           vectors=res['vs'],
                           pareto=testfun['ps'],
                           scale=0.05,
                           arrow_scale=0.4,
                           point_name='x',
                           vector_name='v')

    plot_2D_points_vectors(res['F'],
                           vectors=res['as'],
                           pareto=testfun['pf'],
                           scale=0.5,
                           arrow_scale=0.4,
                           point_name='f(x)',
                           vector_name='a')

