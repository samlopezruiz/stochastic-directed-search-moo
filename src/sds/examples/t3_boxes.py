import os
import time

import numpy as np
import plotly.io as pio
from plotly.subplots import make_subplots

from src.sds.examples.problems.utestfun import TestFuncs
from src.sds.core.continuation import ContinuationBoxes
from src.sds.core.problem import ContinuationProblem
from src.sds.factory import get_corrector, get_tfun, get_predictor
from src.utils.plot import plot_2D_predictor_corrector, plot_boxes_2d, plot_points_centers_2d

pio.renderers.default = "browser"
if __name__ == '__main__':
    # %%
    cfg = {'save_plots': False,
           'problem_name': 't3_ag'
           }

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
    f_limits = np.array([[0., 8.], [0., 5.]])

    predictor = get_predictor('no_adjustment',
                              problem=problem)

    corrector = get_corrector('delta_criteria',
                              problem=problem,
                              t_fun=get_tfun('weighted_dominance',
                                             problem=problem,
                                             eps=1e-5,
                                             maxiter=50),
                              a_fun=lambda a, dx: a,
                              step_eps=5e-1,
                              in_pf_eps=5e-4,
                              maxiter=10)

    ds_cont = ContinuationBoxes(problem=problem,
                                predictor=predictor,
                                corrector=corrector,
                                limits=f_limits,
                                tree_h_coef=0.5,
                                step_eps=5e-1,
                                history=True,
                                )

    t0 = time.time()
    problem.n_f_evals, problem.n_grad_evals = 0, 0
    results = ds_cont.run(x0)
    print('time: {} s'.format(round(time.time() - t0, 4)))
    print('f(x) evals: {}, dx(x) evals: {}'.format(problem.n_f_evals, problem.n_grad_evals))
    print(results['evaluations'])
    # %%
    res = results['population']
    file_path = os.path.join('../img')
    plot_2D_predictor_corrector(points=res['X'],
                                predictors=res['X_p'],
                                correctors=res['X_c'],
                                pareto=testfun['ps'],
                                arrow_scale=0.4,
                                point_name='x',
                                markersize=6,
                                line_width=1.5,
                                plot_arrows=False,
                                plot_points=True,
                                save=cfg['save_plots'],
                                file_path=os.path.join('../img',
                                                       'decision_space_{}'.format(cfg['problem_name'])),
                                )

    fig = make_subplots(rows=1, cols=1)
    pc_fig = plot_2D_predictor_corrector(points=res['F'],
                                         predictors=res['F_p'],
                                         correctors=res['F_c'],
                                         pareto=testfun['pf'],
                                         arrow_scale=0.4,
                                         point_name='f(x)',
                                         markersize=6,
                                         line_width=1.5,
                                         plot_arrows=False,
                                         plot_points=True,
                                         return_fig=True,
                                         save=cfg['save_plots'],
                                         file_path=os.path.join('../img',
                                                                'objective_space_{}'.format(cfg['problem_name'])),
                                         )

    boxes_edges = ds_cont.boxes.get_boxes()
    box_fig = plot_boxes_2d(boxes_edges, return_fig=True)
    fig.add_traces(data=pc_fig.data)
    fig.add_traces(data=box_fig.data)
    fig.show()

    # %%
    points = ds_cont.boxes.get_points()

    pts_fig = plot_points_centers_2d(points['fx'],
                                     centers=points['c'],
                                     best=points['best_ix'],
                                     return_fig=True,
                                     markersize=6)
    fig_data = [box_fig.data, pts_fig.data]
    fig = make_subplots()
    for data in fig_data:
        fig.add_traces(data)

    fig.show()
