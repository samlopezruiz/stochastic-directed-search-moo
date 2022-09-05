import os
import time
from pprint import pprint

import joblib
import numpy as np
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination, get_reference_directions
from pymoo.optimize import minimize

from src.models.d_search.algorithms.continuation import BiDirectionalDsContinuation
from src.models.d_search.utils.plot import plot_bidir_2D_points_vectors, plot_2D_points_traces
from src.models.d_search.utils.factory import get_predictor, get_tfun, get_corrector, get_cont_termination
from src.timeseries.moo.moea.core.problem import TsQuantileProblem
from src.timeseries.utils.continuation import get_q_moo_params_for_problem
from src.timeseries.utils.filename import get_result_folder
from src.timeseries.utils.moo import get_hypervolume

if __name__ == '__main__':
    # %%
    cfg = {'save_plots': True,
           'problem_name': 'ts_q'
           }

    solve_moea = True
    project = 'snp'
    results_cfg = {'experiment_name': '60t_ema_q159',
                   'results': 'TFTModel_ES_ema_r_q159_lr01_pred'}

    results_folder = get_result_folder(results_cfg, project)
    model_results = joblib.load(os.path.join(results_folder, results_cfg['results'] + '.z'))
    model_params = get_q_moo_params_for_problem(project, model_results)

    # %%
    t0 = time.time()
    limits = np.array([1., 1.])
    problem = TsQuantileProblem(y_true=model_params['y_true'],
                                x_data=model_params['valid'],
                                model=model_params['model'],
                                eval_fs=[model_params['loss']],
                                n_obj=2,
                                constraints_limits=limits,
                                quantile_ix=0)
    print('init core time: {}'.format(round(time.time() - t0, 4)))

    # %%
    if solve_moea:
        ref_dirs = get_reference_directions("energy", problem.n_obj, 100)
        algorithm = NSGA3(
            pop_size=100,
            ref_dirs=ref_dirs,
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
                       save_history=False,
                       verbose=True)

        X = res.X
        F = res.F

        ixs = np.argsort(F[:, 0])
        pf = F[ixs]
        ps = X[ixs]

    # %%
    problem.constraints_limits = None
    problem.n_constr = 0
    predictor = get_predictor('no_adjustment', problem=problem)

    corrector = get_corrector('delta_criteria',
                              problem=problem,
                              t_fun=get_tfun('weighted_dominance',
                                             problem=problem,
                                             eps=1e-5,
                                             maxiter=50),
                              a_fun=lambda a, dx: a,
                              step_eps=2e-2,
                              in_pf_eps=1e-5,
                              maxiter=20)

    ds_cont = BiDirectionalDsContinuation(problem=problem,
                                          predictor=predictor,
                                          corrector=corrector,
                                          limits=limits,
                                          step_eps=2e-2,
                                          termination=get_cont_termination('tol', tol=8e-4),
                                          history=True)

    # %%
    t0 = time.time()
    problem.n_f_evals, problem.n_grad_evals = 0, 0
    results = ds_cont.run(np.reshape(problem.original_x, (-1)))
    print('time: {} s'.format(round(time.time() - t0, 4)))
    print('f(x) evals: {}, dx(x) evals: {}'.format(problem.n_f_evals, problem.n_grad_evals))
    pprint(results['evaluations'])

    # %%
    file_path = os.path.join(results_folder, 'img', cfg['problem_name'])
    pareto = {'pf': pf if solve_moea else None, 'ps': ps if solve_moea else None}
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
                                 plot_points=True,
                                 plot_ps=False)

    # %%
    if solve_moea:
        plot_traces = [results['population']['F'], pf]
        plot_2D_points_traces(points_traces=plot_traces,
                              names=['continuation', 'moea'],
                              markersizes=[6, 6],
                              color_ixs=[0, 2],
                              file_path=file_path,
                              size=(1980, 1080))

    # %%
    hv_moea = get_hypervolume(pf, ref=[2., 2.]) if solve_moea else np.nan
    hv_pop = get_hypervolume(results['population']['F'], ref=[2., 2.])
    print('pop hv: {}, moea hv:{}'.format(hv_pop, hv_moea))

    # %%
