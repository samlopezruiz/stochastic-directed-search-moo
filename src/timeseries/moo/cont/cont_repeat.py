import os
import time

import joblib
import numpy as np
from tabulate import tabulate

from src.moo.core.continuation import BiDirectionalDsContinuation
from src.moo.factory import get_corrector, get_predictor, get_tfun, get_cont_termination
from src.timeseries.moo.cont.core.problem import TsQuantileProblem
from src.timeseries.utils.continuation import get_q_moo_params_for_problem
from src.timeseries.utils.filename import get_result_folder
from src.timeseries.utils.files import save_vars

if __name__ == '__main__':
    # %%
    cfg = {'save_plots': False,
           'problem_name': 'ts_q',
           'split_model': "small",
           'quantile_ix': 0,
           }

    project = 'snp'
    results_cfg = {'experiment_name': '60t_ema_q258_5',
                   'results': 'TFTModel_ES_ema_r_q258_lr01_pred'}

    for i in range(7, 15):
        results_cfg['experiment_name'] = '60t_ema_q258_' + str(i)
        results_folder = get_result_folder(results_cfg, project)
        model_results = joblib.load(os.path.join(results_folder, results_cfg['results'] + '.z'))
        model_params = get_q_moo_params_for_problem(project, model_results, shuffle_data=True, random_state=42)

        t0 = time.time()
        limits = np.array([1., 1.])
        problem = TsQuantileProblem(y_train=model_params['y_train'],
                                    x_train=model_params['x_train'],
                                    y_valid=model_params['y_valid'],
                                    x_valid=model_params['x_valid'],
                                    model=model_params['model'].model,
                                    eval_fs=[model_params['loss']],
                                    constraints_limits=limits,
                                    quantile_ix=cfg['quantile_ix'],
                                    base_batch_size=2 ** 11,
                                    moo_batch_size=2 ** 9,
                                    moo_model_size=cfg['split_model'])

        print('init core time: {}'.format(round(time.time() - t0, 4)))

        problem.constraints_limits = None
        problem.n_constr = 0
        predictor = get_predictor('no_adjustment', problem=problem)

        in_pf_eps_cfg = {'small': {'delta': 1e-4, 'rank': 1e-2, 'projection': 5e-4},
                         'medium': {'delta': 1e-4, 'rank': 8e-3, 'projection': 5e-4}}

        corrector_type = 'delta'
        corrector = get_corrector(corrector_type,
                                  problem=problem,
                                  t_fun=get_tfun('weighted_dominance',
                                                 problem=problem,
                                                 eps=1e-4,
                                                 maxiter=50),
                                  a_fun=lambda a, dx: a,
                                  # use_cvxpy=True,
                                  # recalc_v=True,
                                  batch_gradient=True,
                                  mean_grad_stop_criteria=True,
                                  batch_ratio_stop_criteria=0.1,
                                  step_eps=2e-2,
                                  in_pf_eps=in_pf_eps_cfg[cfg['split_model']][corrector_type],  # in_pf_eps=5e-3,
                                  maxiter=30,
                                  )

        ds_cont = BiDirectionalDsContinuation(problem,
                                              predictor,
                                              corrector,
                                              get_cont_termination('tol', tol=8e-4),
                                              limits=limits,
                                              step_eps=2e-2,
                                              verbose=True,
                                              )

        # %%
        t0 = time.time()
        problem.n_f_evals, problem.n_grad_evals = 0, 0
        results = ds_cont.run(np.reshape(problem.original_x, (-1)))
        exec_time = round(time.time() - t0, 2)
        print('time: {} s'.format(exec_time))
        print('f(x) evals: {}, dx(x) evals: {}'.format(problem.n_f_evals, problem.n_grad_evals))

        print(tabulate([[name, *inner.values()] for name, inner in results['evaluations'].items()],
                       tablefmt='psql',
                       headers=list(results['evaluations'][list(results['evaluations'].keys())[0]].keys())))

        cont_filename = corrector_type
        save_vars(results, os.path.join(results_folder, 'cont', cont_filename))
