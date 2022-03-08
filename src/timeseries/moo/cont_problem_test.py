import os
import time

import joblib
import numpy as np

from src.models.attn.nn_funcs import QuantileLossCalculator
from src.timeseries.moo.core.problem import TsQuantileProblem
from src.timeseries.moo.dual_problem_def import DualQuantileWeights
from src.timeseries.utils.continuation import get_q_moo_params_for_problem
from src.timeseries.utils.filename import get_result_folder
from src.timeseries.utils.harness import get_model_data_config
from src.timeseries.utils.moo import get_ix_ind_from_weights

if __name__ == '__main__':
    # %%
    cfg = {'save_plots': False,
           'problem_name': 'ts_q'
           }

    solve_moea = True
    project = 'snp'
    results_cfg = {'experiment_name': '60t_ema_q159',
                   'results': 'TFTModel_ES_ema_r_q159_lr01_pred'}

    model_results = joblib.load(os.path.join(get_result_folder(results_cfg, project), results_cfg['results'] + '.z'))
    model_params = get_q_moo_params_for_problem(project, model_results)

    # %%
    t0 = time.time()
    problem = TsQuantileProblem(y_true=model_params['y_true'],
                                x_data=model_params['valid'],
                                model=model_params['model'],
                                eval_fs=[model_params['loss']],
                                quantile_ix=0)

    print('init core time: {}'.format(round(time.time() - t0, 4)))

    algo_cfg = {'termination': ('n_gen', 100),
                'pop_size': 100,
                'use_sampling': True,
                'optimize_eq_weights': False,
                'use_constraints': True,
                'constraints': [1., 1.],
                }

    moo_method = 'NSGA2'

    model_results = joblib.load(os.path.join(get_result_folder(results_cfg, project), results_cfg['results'] + '.z'))

    config, data_formatter, model_folder = get_model_data_config(project,
                                                                 model_results['experiment_cfg'],
                                                                 model_results['model_params'],
                                                                 model_results['fixed_params'])
    experiment_cfg = model_results['experiment_cfg']

    dual_q_problem = DualQuantileWeights(architecture=experiment_cfg['architecture'],
                                         model_folder=model_folder,
                                         data_formatter=data_formatter,
                                         data_config=config.data_config,
                                         use_gpu=True,
                                         parallelize_pop=False if moo_method == 'MOEAD' else True,
                                         constraints_limits=algo_cfg['constraints'] if algo_cfg[
                                             'use_constraints'] else None,
                                         optimize_eq_weights=algo_cfg['optimize_eq_weights'])

    lower_q_problem, upper_q_problem = dual_q_problem.get_problems()

    #%%
    x0 = problem.original_x
    f0 = problem.evaluate(x0)
    print(f0)
    dx = problem.gradient(x0)

    weights = lower_q_problem.original_weights
    ind = get_ix_ind_from_weights(weights, lower_q_problem.ix_weight)
    f1 = lower_q_problem.evaluate(ind)
    print(f1)

    #%%
    quantile_loss_per_q_moo = QuantileLossCalculator(model_params['model'].quantiles, model_params['model'].output_size).quantile_loss_per_q_moo
    numpy_quantile_loss_per_q_moo = QuantileLossCalculator(model_params['model'].quantiles, model_params['model'].output_size).numpy_quantile_loss_per_q_moo

    #%%=
    y_pred = problem.predict()
    my_map = lower_q_problem.get_pred_func(ind)
    y_pred1 = np.squeeze(my_map().numpy())
    y_true = problem.y_true

    loss0 = quantile_loss_per_q_moo(y_true, y_pred)
    # print([tf.reduce_mean(l, axis=-1).numpy() for l in loss0[problem.quantile_ix]])
    print(loss0)
    loss1 = np.array(numpy_quantile_loss_per_q_moo(y_true, y_pred))
    print(loss1)