import os
import time
from pprint import pprint

import numpy as np
import plotly.io as pio
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import ReferenceDirectionSurvival
from pymoo.core.population import pop_from_array_or_individual
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_problem, get_reference_directions
from pymoo.factory import get_termination
from pymoo.optimize import minimize

from src.moo.examples.problems.utestfun import TestFuncs
from src.moo.core.continuation import ContinuationBoxes
from src.moo.core.problem import AutomaticDifferentiationProblem
from src.moo.factory import get_tfun, get_corrector, get_predictor
from src.moo.utils.indicators import hypervolume
from src.utils.plot import plot_boxes_3d, plot_points_3d, plot_traces

pio.renderers.default = "browser"
if __name__ == '__main__':
    # %%
    cfg = {'save_plots': False,
           'problem_name': 'zdt2'
           }

    solve_moea = True
    plot_boxes = False

    tfun = TestFuncs()
    testfun = tfun.get(cfg['problem_name'])
    problem = get_problem("dtlz2", n_var=30, n_obj=3)
    problem = AutomaticDifferentiationProblem(problem)

    # %%
    if solve_moea:
        algorithm = NSGA2(
            pop_size=750,
            n_offsprings=750,
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
    x0 = np.array([0.38726374, 0.50142701, 0.48666312, 0.5119315, 0.50954044, 0.4951654,
                   0.50899326, 0.51179125, 0.48861775, 0.52403041, 0.47975356, 0.52147994,
                   0.46698674, 0.50230151, 0.50465705, 0.51797048, 0.50098518, 0.48723901,
                   0.5261419, 0.51706457, 0.47990969, 0.49896529, 0.47078924, 0.48367379,
                   0.50657368, 0.48315766, 0.47689119, 0.52006725, 0.50089009, 0.49232051])
    f0 = problem.evaluate(x0)

    f_limits = np.array([[0., 1.]] * 3)

    predictor = get_predictor('no_adjustment',
                              problem=problem,
                              eps=1e-2)

    corrector = get_corrector('delta_criteria',
                              problem=problem,
                              t_fun=get_tfun('weighted_dominance',
                                             problem=problem,
                                             eps=1e-5,
                                             maxiter=50),
                              a_fun=lambda a, dx: a,
                              step_eps=9e-2,
                              in_pf_eps=1e-2,
                              maxiter=100)

    ds_cont = ContinuationBoxes(problem=problem,
                                predictor=predictor,
                                corrector=corrector,
                                f_limits=f_limits,
                                c=0.8,
                                # h_max=14,
                                step_eps=9e-2,
                                debug=True,
                                history=True,
                                )

    t0 = time.time()
    problem.n_f_evals, problem.n_grad_evals = 0, 0
    results = ds_cont.run(x0)
    print('time: {} s'.format(round(time.time() - t0, 4)))
    print('f(x) evals: {}, dx(x) evals: {}'.format(problem.n_f_evals, problem.n_grad_evals))
    pprint(results['evaluations'])

    # %% Hypervolume
    points = ds_cont.boxes.get_points()
    print('{} solutions found, {} best solutions found'.format(len(points['fx']),
                                                               len(points['fx'][points['best_ix'], :])))
    hv_moea = hypervolume(pf_moea, ref=[2., 2., 2.]) if solve_moea else np.nan
    hv_pop = hypervolume(results['population']['F'], ref=[2., 2., 2.])
    hv_ref = hypervolume(testfun['pf'], ref=[2., 2., 2.])
    print('ref hv: {}, pop hv: {}, moea hv:{}'.format(hv_ref, hv_pop, hv_moea))

    # %%
    file_path = os.path.join('../img')

    if plot_boxes:
        boxes_edges = ds_cont.boxes.get_boxes()
        box_fig = plot_boxes_3d(boxes_edges, return_fig=True)

        pts_fig = plot_points_3d(points['fx'],
                                 secondary=points['c'],
                                 mask=points['best_ix'],
                                 return_fig=True,
                                 markersize=5)

        plot_traces([box_fig.data, pts_fig.data])

    plot_points_3d(points['fx'],
                   secondary=points['c'],
                   mask=points['best_ix'],
                   markersize=5,
                   only_best=False,
                   title='Continuation method Pareto Front')

    if solve_moea:
        plot_points_3d(pf_moea, markersize=7, title='MOEA method Pareto Front')

    # %%
    F = points['fx']
    ref_dirs = get_reference_directions("energy", problem.n_obj, 500)
    pop = pop_from_array_or_individual(F)
    pop.set('F', F)
    rds = ReferenceDirectionSurvival(ref_dirs)
    niching = rds.do(problem, pop, n_survive=500)
    opt = rds.opt.get('F')
    plot_points_3d(opt,
                   markersize=8,
                   title='Continuation method Pareto Front with nitching')
