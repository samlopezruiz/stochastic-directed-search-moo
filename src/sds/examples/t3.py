import os
import time
from pprint import pprint

import numpy as np
import plotly.io as pio
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.factory import get_termination
from pymoo.optimize import minimize

from src.sds.examples.problems.utestfun import TestFuncs
from src.sds.core.continuation import DsContinuation
from src.sds.core.problem import ContinuationProblem
from src.sds.utils.indicators import hypervolume
from src.utils.plot import plot_2D_points_vectors, plot_2D_predictor_corrector, plot_2D_points_traces
from src.sds.factory import get_tfun, get_corrector, get_predictor, get_cont_termination

pio.renderers.default = "browser"
if __name__ == '__main__':
    # %%
    cfg = {'save_plots': False,
           'problem_name': 't3_ag'
           }

    solve_moea = False

    tfun = TestFuncs()
    testfun = tfun.get(cfg['problem_name'])
    problem = ContinuationProblem(f=testfun['f'],
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

    corrector = get_corrector('ds',
                              problem=problem,
                              t_fun=get_tfun('weighted_dominance',
                                             problem=problem,
                                             eps=1e-5,
                                             maxiter=50),
                              a_fun=lambda a, dx: a,
                              step_eps=5e-1,
                              in_pf_eps=5e-4,
                              maxiter=10)

    ds_cont = DsContinuation(problem=problem,
                             predictor=predictor,
                             corrector=corrector,
                             dir=1,
                             step_eps=5e-1,
                             termination=get_cont_termination('tol', tol=5e-3),
                             history=True,
                             )

    t0 = time.time()
    problem.n_f_evals, problem.n_grad_evals = 0, 0
    results = ds_cont.run(x0)
    print('time: {} s'.format(round(time.time() - t0, 4)))
    print('f(x) evals: {}, dx(x) evals: {}'.format(problem.n_f_evals, problem.n_grad_evals))
    pprint(results['evaluations'])

    # %%
    file_path = os.path.join('../img')
    res = results['population']
    plot_2D_predictor_corrector(points=res['X'],
                                predictors=res['X_p'],
                                correctors=res['X_c'],
                                pareto=testfun['ps'],
                                title='Continuation method: predictors and correctors in decision space',
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
                                title='Continuation method: predictors and correctors in objective space',
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
                           title='Continuation method: movement vector (v) decision space',
                           scale=0.05,
                           arrow_scale=0.4,
                           point_name='x',
                           vector_name='v')

    plot_2D_points_vectors(res['F'],
                           vectors=res['as'],
                           pareto=testfun['pf'],
                           title='Continuation method: normal vector (a) to PF in objective space',
                           scale=0.5,
                           arrow_scale=0.4,
                           point_name='f(x)',
                           vector_name='a')

    # %%
    if solve_moea:
        plot_traces = [testfun['pf'], pf_moea, results['population']['F']]
        plot_2D_points_traces(points_traces=plot_traces,
                              names=['reference', 'moea', 'continuation'],
                              modes=['lines', 'markers', 'markers'],
                              title='Soultions comparison in Pareto front',
                              markersizes=[6, 6, 8],
                              color_ixs=[10, 0, 3],
                              file_path=file_path,
                              size=(1980, 1080))

    # %%
    hv_ref = hypervolume(testfun['pf'], ref=[20., 20.]) if testfun['pf'] is not None else np.nan
    hv_moea = hypervolume(pf_moea, ref=[20., 20.]) if solve_moea else np.nan
    hv_pop = hypervolume(results['population']['F'], ref=[20., 20.])
    print('ref hv: {}, sds hv: {}, moea hv:{}'.format(hv_ref, hv_pop, hv_moea))
