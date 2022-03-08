import copy
import os
import time

import joblib
import numpy as np

from src.models.attn.hyperparam_opt import HyperparamOptManager
from src.models.attn.nn_funcs import QuantileLossCalculator
from src.timeseries.moo.core.problem import TsQuantileProblem
from src.timeseries.utils.filename import get_result_folder
from src.timeseries.utils.harness import get_model_data_config, get_model

if __name__ == '__main__':
    # %%
    project = 'snp'
    results_cfg = {'experiment_name': '60t_ema_q159',
                   'results': 'TFTModel_ES_ema_r_q159_lr01_pred'
                   }

    model_results = joblib.load(os.path.join(get_result_folder(results_cfg, project), results_cfg['results'] + '.z'))

    config, data_formatter, model_folder = get_model_data_config(project,
                                                                 model_results['experiment_cfg'],
                                                                 model_results['model_params'],
                                                                 model_results['fixed_params'])
    experiment_cfg = model_results['experiment_cfg']

    Model = get_model(experiment_cfg['architecture'])
    train, valid, test = data_formatter.split_data(config.data_config)

    # Sets up default params
    fixed_params = data_formatter.get_experiment_params()
    params = data_formatter.get_default_model_params()
    params["model_folder"] = model_folder

    # Sets up hyperparam manager
    opt_manager = HyperparamOptManager({k: [params[k]] for k in params}, fixed_params, model_folder)
    model_params = opt_manager.get_next_parameters()

    model = Model(model_params)
    model.load(opt_manager.hyperparam_folder, use_keras_loadings=True)

    outputs, output_map, data_map = model.predict_all(valid, batch_size=128)
    pre_last_layer_output = outputs['transformer_output']

    quantiles = copy.copy(model.quantiles)
    output_size = copy.copy(model.output_size)

    quantile_loss = QuantileLossCalculator(quantiles, output_size).quantile_loss_per_q_moo
    targets = data_map['outputs']

    y_true = np.concatenate([targets, targets, targets], axis=-1)

    # %%
    t0 = time.time()
    problem = TsQuantileProblem(y_true=y_true,
                                x_data=valid,
                                model=model,
                                eval_fs=[quantile_loss],
                                quantile_ix=0
                                )

    print('init core time: {}'.format(round(time.time() - t0, 4)))

    t0 = time.time()
    x = problem.original_x
    fx = problem.evaluate(x)
    print('eval f(x) time: {}'.format(round(time.time() - t0, 4)))

    t0 = time.time()
    dx = problem.gradient(x)
    print('grad df(x) time: {}'.format(round(time.time() - t0, 4)))

    # %%
