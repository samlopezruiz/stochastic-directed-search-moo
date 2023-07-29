import os
import time

import joblib
import numpy as np
from numpy.linalg import pinv

from tabulate import tabulate

from src.models.attn.hyperparam_opt import HyperparamOptManager
from src.moo.core.continuation import BiDirectionalDsContinuation
from src.moo.factory import get_corrector, get_predictor, get_tfun, get_cont_termination
from src.moo.utils.util import subroutine_times_problem
from src.timeseries.moo.cont.core.config import cont_cfg
from src.timeseries.moo.cont.core.problem import TsQuantileProblem
from src.timeseries.utils.continuation import get_q_moo_params_for_problem, get_q_moo_params_for_problem2
from src.timeseries.utils.filename import get_result_folder
from src.timeseries.utils.files import save_vars
from src.timeseries.utils.harness import get_model_data_config, get_model
from src.timeseries.utils.moo import get_hypervolume, sort_1st_col
from src.utils.plot import plot_bidir_2D_points_vectors, plot_2D_points_traces, plot_2D_points_traces_total

if __name__ == '__main__':
    # %%
    cfg = {'save_plots': False,
           'save_results': False,
           'solve_moea': False,
           }

    project = 'snp'
    cont_cfg['model']['ix'] = 1

    cont_cfg['model']['experiment_name'] = cont_cfg['model']['basename'] + '_' + str(cont_cfg['model']['ix'])
    results_folder = get_result_folder(cont_cfg['model'], project)
    model_results = joblib.load(os.path.join(results_folder, cont_cfg['model']['results'] + '.z'))
    model_params = get_q_moo_params_for_problem2(project,
                                                 model_results,
                                                 shuffle_data=cont_cfg['data']['shuffle'],
                                                 random_state=cont_cfg['data']['random_state'])

    # model_params['opt_manager'].hyperparam_folder = model_params['opt_manager'].hyperparam_folder[:-1] + str(model_ix)
    # model_params['model'].load(model_params['opt_manager'].hyperparam_folder, use_keras_loadings=True)

    # %%
    t0 = time.time()
    problem = TsQuantileProblem(y_train=model_params['datasets']['train']['y'],
                                x_train=model_params['datasets']['train']['x'],
                                y_valid=model_params['datasets']['valid']['y'],
                                x_valid=model_params['datasets']['valid']['x'],
                                model=model_params['model'].model,
                                eval_fs=[model_params['loss']],
                                constraints_limits=cont_cfg['problem']['limits'],
                                quantile_ix=cont_cfg['problem']['quantile_ix'],
                                base_batch_size=cont_cfg['problem']['base_batch_size'],
                                moo_batch_size=cont_cfg['problem']['moo_batch_size'],
                                moo_model_size=cont_cfg['problem']['split_model'])

    print('init core time: {}'.format(round(time.time() - t0, 4)))
    subroutine_times_problem(problem)

    problem.constraints_limits = None
    problem.n_constr = 0
    predictor = get_predictor('no_adjustment', problem=problem)

    corrector = get_corrector(cont_cfg['corrector']['type'],
                              problem=problem,
                              t_fun=get_tfun(cont_cfg['corrector']['t_fun']['type'],
                                             problem=problem,
                                             eps=cont_cfg['corrector']['t_fun']['eps'],
                                             maxiter=cont_cfg['corrector']['t_fun']['maxiter']),
                              a_fun=lambda a, dx: a,
                              batch_gradient=cont_cfg['corrector']['batch_gradient'],
                              mean_grad_stop_criteria=cont_cfg['corrector']['mean_grad_stop_criteria'],
                              batch_ratio_stop_criteria=cont_cfg['corrector']['batch_ratio_stop_criteria'],
                              step_eps=cont_cfg['corrector']['step_eps'],
                              in_pf_eps=cont_cfg['corrector']['in_pf_eps_cfg'][cont_cfg['problem']['split_model']][
                                  cont_cfg['corrector']['type']],
                              maxiter=cont_cfg['corrector']['maxiter']
                              )

    ds_cont = BiDirectionalDsContinuation(problem,
                                          predictor,
                                          corrector,
                                          get_cont_termination(cont_cfg['cont']['termination']['type'],
                                                               cont_cfg['cont']['termination']['thold']),
                                          limits=cont_cfg['problem']['limits'],
                                          step_eps=cont_cfg['cont']['step_eps'],
                                          verbose=cont_cfg['cont']['verbose'],
                                          )

    # Run problem
    t0 = time.time()
    problem.n_f_evals, problem.n_grad_evals = 0, 0
    results = ds_cont.run(np.reshape(problem.original_x, (-1)))
    exec_time = round(time.time() - t0, 2)
    print('time: {} s'.format(exec_time))
    print('f(x) evals: {}, dx(x) evals: {}'.format(problem.n_f_evals, problem.n_grad_evals))

    print(tabulate([[name, *inner.values()] for name, inner in results['evaluations'].items()],
                   tablefmt='psql',
                   headers=list(results['evaluations'][list(results['evaluations'].keys())[0]].keys())))

    hv_train = get_hypervolume(results['population']['F'], ref=[2., 2.])

    hvs = {}
    for hv, key in zip([hv_train], ['train/cont']):
        hvs[key] = hv

    print(tabulate(hvs.items(),
                   headers=['subset/method', 'hypervolume'],
                   tablefmt='psql',
                   floatfmt=(None, ",.6f"),
                   stralign="right"))

    # Save results
    cont_filename = cont_cfg['corrector']['type'] + '_' + cont_cfg['cont']['termination']['type']
    if cfg['save_results']:
        save_vars(results, os.path.join(results_folder, 'cont', cont_filename))

    # Plot results
    file_path = os.path.join(results_folder, 'cont', 'img', cont_filename)
    train_population = [res['population'] for res in results['independent']]
    descent_pops = [res['descent'] for res in results['independent']]

    title = 'tile'
    plot_bidir_2D_points_vectors(train_population,
                                 pareto=None,
                                 descent=descent_pops,
                                 arrow_scale=0.4,
                                 markersize=5,
                                 pareto_marker_mode='markers+lines',
                                 save=cfg['save_plots'],
                                 save_png=False,
                                 file_path=file_path,
                                 size=(1980, 1080),
                                 plot_arrows=True,
                                 plot_points=True,
                                 plot_ps=False,
                                 return_fig=False,
                                 titles=[title, title])

    X_sorted, F_sorted = sort_1st_col(results['population']['X'], results['population']['F'])

    plot_2D_points_traces_total([F_sorted],
                                names=['cont'],
                                markersizes=[5],
                                modes=['markers+lines'])

    #%%
