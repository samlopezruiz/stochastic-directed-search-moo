import os
import time
from pprint import pprint

import plotly.io as pio

from src.moo.examples.problems.utestfun import TestFuncs
from src.moo.core.continuation import BiDirectionalDsContinuation
from src.moo.core.problem import ContinuationProblem
from src.utils.plot import plot_bidir_2D_points_vectors
from src.moo.factory import get_tfun, get_corrector, get_predictor, get_cont_termination

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
    x0 = [-0.10360, -0.48965]
    x0 = [-0., -0.]  # not in PF

    predictor = get_predictor('no_adjustment', problem=problem)

    corrector = get_corrector('delta_criteria',
                              problem=problem,
                              t_fun=get_tfun('weighted_dominance',
                                             problem=problem,
                                             eps=1e-5,
                                             maxiter=10),
                              a_fun=lambda a, dx: a,
                              step_eps=5e-1,
                              in_pf_eps=1e-3,
                              maxiter=10)

    ds_cont = BiDirectionalDsContinuation(problem=problem,
                                          predictor=predictor,
                                          corrector=corrector,
                                          step_eps=5e-1,
                                          termination=get_cont_termination('tol', tol=1e-3),
                                          history=True,
                                          )

    t0 = time.time()
    problem.n_f_evals, problem.n_grad_evals = 0, 0
    results = ds_cont.run(x0)
    print('time: {} s'.format(round(time.time() - t0, 4)))
    print('f(x) evals: {}, dx(x) evals: {}'.format(problem.n_f_evals, problem.n_grad_evals))
    pprint(results['evaluations'])

    # %%
    file_path = os.path.join('../img', cfg['problem_name'])
    pareto = {'pf': testfun['pf'], 'ps': testfun['ps']}
    plot_populations = [res['population'] for res in results['independent']]
    plot_bidir_2D_points_vectors(plot_populations,
                                 pareto,
                                 arrow_scale=0.4,
                                 markersize=5,
                                 save=cfg['save_plots'],
                                 save_png=False,
                                 file_path=file_path,
                                 size=(1980, 1080),
                                 plot_arrows=True,
                                 plot_points=True)
