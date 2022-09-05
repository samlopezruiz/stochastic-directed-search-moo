import tensorflow as tf

import copy

import numpy as np

from src.models.attn.hyperparam_opt import HyperparamOptManager
from src.models.attn.nn_funcs import QuantileLossCalculator
from src.timeseries.utils.harness import get_model_data_config, get_model
from sklearn.utils import shuffle


def suffle_dataset(datasets, random_state):
    x_train, y_train = shuffle(datasets['train']['x'], datasets['train']['y'], random_state=random_state)
    x_valid, y_valid = shuffle(datasets['valid']['x'], datasets['valid']['y'], random_state=random_state)
    x_test, y_test = shuffle(datasets['test']['x'], datasets['test']['y'], random_state=random_state)
    return {'train': {'x': x_train, 'y': y_train},
            'valid': {'x': x_valid, 'y': y_valid},
            'test': {'x': x_test, 'y': y_test}}


def batch_dataset_from_model(data, model, shuffle_data=True, random_state=None):
    train, valid, test = data
    x_train, y_train = get_xy_from_model(model, train)
    x_valid, y_valid = get_xy_from_model(model, valid)
    x_test, y_test = get_xy_from_model(model, test)
    # pre_last_layer_output = outputs['transformer_output']

    if shuffle_data:
        x_train, y_train = shuffle(x_train, y_train, random_state=random_state)
        x_valid, y_valid = shuffle(x_valid, y_valid, random_state=random_state)
        x_test, y_test = shuffle(x_test, y_test, random_state=random_state)

    return {'train': {'x': x_train, 'y': y_train},
            'valid': {'x': x_valid, 'y': y_valid},
            'test': {'x': x_test, 'y': y_test}}


def get_model_and_data_from_config(architecture, data_formatter, config, model_folder):
    train, valid, test = data_formatter.split_data(config.data_config)

    Model = get_model(architecture)
    # Sets up default params
    fixed_params = data_formatter.get_experiment_params()
    params = data_formatter.get_default_model_params()
    params["model_folder"] = model_folder

    # Sets up hyperparameter manager
    opt_manager = HyperparamOptManager({k: [params[k]] for k in params}, fixed_params, model_folder)
    model_params = opt_manager.get_next_parameters()

    model = Model(model_params)
    model.load(opt_manager.hyperparam_folder, use_keras_loadings=True)

    return model, (train, valid, test), opt_manager


def get_quantile_loss_from_model(model):
    quantiles = copy.copy(model.quantiles)
    output_size = copy.copy(model.output_size)

    quantile_loss = QuantileLossCalculator(quantiles, output_size).quantile_loss_per_q_moo
    return quantile_loss


def get_q_moo_params_for_problem2(project, model_results, shuffle_data=True, random_state=None, use_gpu=True):
    with tf.device('/device:GPU:0' if use_gpu else "/cpu:0"):
        config, data_formatter, model_folder = get_model_data_config(project,
                                                                     model_results['experiment_cfg'],
                                                                     model_results['model_params'],
                                                                     model_results['fixed_params'])

        model, data, opt_manager = get_model_and_data_from_config(model_results['experiment_cfg']['architecture'],
                                                                  data_formatter,
                                                                  config,
                                                                  model_folder)

        datasets = batch_dataset_from_model(data,
                                            model,
                                            shuffle_data,
                                            random_state)

        quantile_loss = get_quantile_loss_from_model(model)

    if shuffle_data:
        datasets = suffle_dataset(datasets, random_state)

    return {'datasets': datasets,
            'model': model,
            'loss': quantile_loss,
            'opt_manager': opt_manager}


def get_xy_from_model(model, x):
    data = model._batch_data(x)
    targets = data['outputs']
    y_true = np.concatenate([targets, targets, targets], axis=-1)
    x_data = data['inputs']
    return x_data, y_true


def get_q_moo_params_for_problem(project, model_results, shuffle_data=True, random_state=None):
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

    # Sets up hyperparameter manager
    opt_manager = HyperparamOptManager({k: [params[k]] for k in params}, fixed_params, model_folder)
    model_params = opt_manager.get_next_parameters()

    model = Model(model_params)
    model.load(opt_manager.hyperparam_folder, use_keras_loadings=True)

    x_train, y_train = get_xy_from_model(model, train)
    x_valid, y_valid = get_xy_from_model(model, valid)
    x_test, y_test = get_xy_from_model(model, test)
    # pre_last_layer_output = outputs['transformer_output']

    if shuffle_data:
        x_train, y_train = shuffle(x_train, y_train, random_state=random_state)
        x_valid, y_valid = shuffle(x_valid, y_valid, random_state=random_state)
        x_test, y_test = shuffle(x_test, y_test, random_state=random_state)

    quantiles = copy.copy(model.quantiles)
    output_size = copy.copy(model.output_size)

    quantile_loss = QuantileLossCalculator(quantiles, output_size).quantile_loss_per_q_moo
    # targets = data_map['outputs']

    return {'y_train': y_train,
            'x_train': x_train,
            'x_valid': x_valid,
            'y_valid': y_valid,
            'model': model,
            'loss': quantile_loss,
            'opt_manager': opt_manager}


def get_xy_from_model(model, x):
    data = model._batch_data(x)
    targets = data['outputs']
    y_true = np.concatenate([targets, targets, targets], axis=-1)
    x_data = data['inputs']
    return x_data, y_true
