import copy

import numpy as np

from src.models.attn.hyperparam_opt import HyperparamOptManager
from src.models.attn.nn_funcs import QuantileLossCalculator
from src.timeseries.utils.harness import get_model_data_config, get_model


def get_q_moo_params_for_problem(project, model_results):
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

    # outputs, output_map, data_map = model.predict_all(valid, batch_size=128)
    targets = model._batch_data(valid)['outputs']
    y_true = np.concatenate([targets, targets, targets], axis=-1)
    # pre_last_layer_output = outputs['transformer_output']

    quantiles = copy.copy(model.quantiles)
    output_size = copy.copy(model.output_size)

    quantile_loss = QuantileLossCalculator(quantiles, output_size).quantile_loss_per_q_moo
    # targets = data_map['outputs']

    return {'y_true': y_true, 'valid': valid, 'model': model, 'loss': quantile_loss}
