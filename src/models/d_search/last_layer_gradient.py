import copy
import os

import joblib
import numpy as np
import seaborn as sns
import telegram_send
import tensorflow as tf
from src.models.attn.hyperparam_opt import HyperparamOptManager
from src.models.attn.nn_funcs import QuantileLossCalculator
from src.models.attn.utils import extract_numerical_data
from src.timeseries.moo.dual_problem_def import DualQuantileWeights
from src.timeseries.utils.filename import get_result_folder, quantiles_name, termination_name
from src.timeseries.utils.files import save_vars
from src.timeseries.utils.harness import get_model_data_config, get_model
from src.timeseries.utils.moo import get_last_layer_weights, create_output_map, compute_moo_q_loss, moo_q_loss_model
from src.timeseries.utils.moo_harness import run_dual_moo_weights
from src.models.attn import utils

sns.set_theme('poster')

if __name__ == '__main__':
    # %%
    general_cfg = {'save_results': True,
                   'save_history': True,
                   'send_notifications': True}

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
    weights, last_layer = get_last_layer_weights(model)

    #%%
    outputs, output_map, data_map = model.predict_all(valid, batch_size=128)
    pre_last_layer_output = outputs['transformer_output']

    quantiles = copy.copy(model.quantiles)
    output_size = copy.copy(model.output_size)

    quantile_loss = QuantileLossCalculator(quantiles, output_size).quantile_loss
    targets = data_map['outputs']

    #%%
    y_true = np.concatenate([targets, targets, targets], axis=-1)

    with tf.GradientTape() as tape:
        # Forward pass
        y_pred = last_layer(pre_last_layer_output)
        loss = quantile_loss(y_true, y_pred)

    grad = tape.gradient(loss, last_layer.trainable_variables)

    for var, g in zip(last_layer.trainable_variables, grad):
        print(f'{var.name}, shape: {g.shape}')

