import os
import time
from collections import defaultdict
from copy import deepcopy

import joblib
import numpy as np
import pandas as pd
from numpy.linalg import pinv, matrix_rank
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination, get_reference_directions
from pymoo.optimize import minimize
from tabulate import tabulate

from src.moo.core.continuation import BiDirectionalDsContinuation
from src.moo.factory import get_corrector, get_predictor, get_tfun, get_cont_termination
from src.timeseries.moo.cont.core.problem import TsQuantileProblem
from src.moo.nn.problem_old import GradientTestsProblem
from src.moo.nn.utils import batch_array, get_one_output_model, split_model, reconstruct_weights, \
    params_conversion_weights, predict_from_batches, batch_from_list_or_array
from src.moo.utils.functions import in_pareto_front
from src.timeseries.utils.continuation import get_q_moo_params_for_problem
from src.timeseries.utils.filename import get_result_folder
from src.timeseries.utils.moo import get_hypervolume
from src.utils.plot import plot_bidir_2D_points_vectors, plot_2D_points_vectors, plot_traces

if __name__ == '__main__':
    # %%
    cfg = {'save_plots': False,
           'problem_name': 'ts_q'
           }

    solve_moea = False
    project = 'snp'
    results_cfg = {'experiment_name': '60t_ema_q159',
                   'results': 'TFTModel_ES_ema_r_q159_lr01_pred'}

    results_folder = get_result_folder(results_cfg, project)
    model_results = joblib.load(os.path.join(results_folder, results_cfg['results'] + '.z'))
    model_params = get_q_moo_params_for_problem(project, model_results, shuffle_data=True)

    # %%
    t0 = time.time()
    limits = np.array([1., 1.])
    # problem = TsQuantileProblem(y_train=model_params['y_train'],
    #                             x_train=model_params['x_train'],
    #                             y_valid=model_params['y_valid'],
    #                             x_valid=model_params['x_valid'],
    #                             model=model_params['model'].model,
    #                             eval_fs=[model_params['loss']],
    #                             n_obj=2,
    #                             quantile_ix=0,
    #                             base_batch_size=2 ** 8,
    #                             moo_batch_size=2 ** 8,
    #                             moo_model_size='small')

    print('init core time: {}'.format(round(time.time() - t0, 4)))

    # %%e
    # x0_medium = problem.original_x
    #
    # t0 = time.time()
    # f0_medium = problem.evaluate(problem.original_x)
    # print('eval time: {:.4} s'.format(time.time() - t0))
    #
    # t0 = time.time()
    # dx_medium = problem.gradient(problem.original_x)
    # print('grad time: {:.4} s'.format(time.time() - t0))
    #
    # t0 = time.time()
    # dx_1 = pinv(dx_medium)
    # print('inverse time: {:.4} s'.format(time.time() - t0))
    #
    # print(f0_medium)

    # %%
    print('original losses')
    y_train = model_params['y_train']
    model = get_one_output_model(model_params['model'].model, output_layer_name='td_quantiles')
    x_train_batches = batch_from_list_or_array(model_params['x_train'], batch_size=256)

    y_pred = predict_from_batches(model,
                                  x_train_batches,
                                  to_numpy=False,
                                  concat_output=True)
    loss = model_params['loss'](y_train, y_pred)
    losses_model = [l.numpy() for l in loss[0]]
    print('losses_model', losses_model)

    # %%
    # print('small moo losses')
    # intermediate_layers = ['layer_normalization_40']
    # models = split_model(model, intermediate_layers)
    # moo_model, base_model = models['trainable_model'], models['base_model']
    #
    # moo_model_input_train = predict_from_batches(base_model, x_train_batches,
    #                                              to_numpy=False,
    #                                              concat_output=False)
    #
    # trainable_weights = moo_model.trainable_weights
    # # weight labels contain only the layer they correspond
    # # kernel and bias labels are not included
    # weights, weights_lbls = [w.numpy() for w in trainable_weights], [w.name.split('/')[0] for w in trainable_weights]
    # individual, ind_weights_params = params_conversion_weights(weights)
    #
    # weights = reconstruct_weights(individual, ind_weights_params)
    #
    # weights_dict = defaultdict(list)
    #
    # # weights are represented as lists with the layer's name as key
    # for lbl, w in zip(weights_lbls, weights):
    #     weights_dict[lbl].append(w)
    #
    # # replace weights of layers with trainable weights
    # for layer in moo_model.layers:
    #     if layer.name in weights_dict:
    #         layer.set_weights(weights_dict[layer.name])
    #
    # y_pred = predict_from_batches(moo_model,
    #                               moo_model_input_train,
    #                               to_numpy=False,
    #                               concat_output=True)
    # loss = model_params['loss'](y_train, y_pred)
    # losses_moo_model = [l.numpy() for l in loss[0]]
    #
    # print('losses_small_moo_model', losses_moo_model)

    # %%
    import tensorflow as tf
    from tensorflow.keras.models import clone_model


    def split_model(model, intermediate_layers):
        base_model0 = tf.keras.Model(inputs=model.inputs,
                                     outputs=[model.get_layer(l).output for l in
                                              intermediate_layers])
        base_model = clone_model(base_model0)
        base_model.set_weights(base_model0.get_weights())

        trainable_model0 = tf.keras.Model(inputs=[model.get_layer(l).output for l in intermediate_layers],
                                          outputs=model.outputs)
        trainable_model = clone_model(trainable_model0)
        trainable_model.set_weights(trainable_model0.get_weights())

        # base_model.compile()
        # trainable_model.compile()
        return {'base_model': base_model,
                'trainable_model': trainable_model}  # %%


    print('medium moo losses')
    model = get_one_output_model(model_params['model'].model, output_layer_name='td_quantiles')
    layers = model.layers
    intermediate_layers = ['layer_normalization_36', 'time_distributed_144']
    models = split_model(model, intermediate_layers)
    moo_model, base_model = models['trainable_model'], models['base_model']

    # base_model.trainable = False
    # moo_model.trainable = False
    print('-> moo_model_input_train...')
    moo_model_input_train = predict_from_batches(base_model, x_train_batches,
                                                 to_numpy=False,
                                                 concat_output=False)
    print(moo_model.inputs)
    print([bat.shape for bat in moo_model_input_train[0]])

    # moo_model.compile()
    trainable_weights = moo_model.get_weights()
    # moo_model.set_weights(trainable_weights)
    print('trainable: {}, non-trainable: {}'.format(len(moo_model.trainable_weights),
                                                    len(moo_model.non_trainable_weights)))

    # weight labels contain only the layer they correspond
    # kernel and bias labels are not included
    # weights, weights_lbls = [w.numpy() for w in trainable_weights], [w.name.split('/')[0] for w in trainable_weights]

    individual, ind_weights_params = params_conversion_weights(trainable_weights)
    weights = reconstruct_weights(individual, ind_weights_params)
    # individual0, _ = params_conversion_weights(weights)
    # print(np.sum(np.abs(individual - individual0)))

    moo_model.set_weights(weights)
    #
    # weights_dict = defaultdict(list)
    #
    # # weights are represented as lists with the layer's name as key
    # for lbl, w in zip(weights_lbls, weights):
    #     weights_dict[lbl].append(w)
    #
    # # replace weights of layers with trainable weights
    # for layer in moo_model.layers:
    #     if layer.name in weights_dict:
    #         layer.set_weights(weights_dict[layer.name])

    y_pred = predict_from_batches(moo_model,
                                  moo_model_input_train,
                                  to_numpy=False,
                                  concat_output=True)
    loss = model_params['loss'](y_train, y_pred)
    losses_moo_model = [l.numpy() for l in loss[0]]

    print('losses_medium_moo_model', losses_moo_model)

    # %%
    # t0 = time.time()
    # limits = np.array([1., 1.])
    # problem_small = TsQuantileProblem(y_train=mod                                  el_params['y_train'],
    #                                   x_train=model_params['x_train'],
    #                                   y_valid=model_params['y_valid'],
    #                                   x_valid=model_params['x_valid'],
    #                                   model=model_params['model'].model,
    #                                   eval_fs=[model_params['loss']],
    #                                   n_obj=2,
    #                                   quantile_ix=0,
    #                                   base_batch_size=2 ** 8,
    #                                   moo_batch_size=2 ** 8,
    #                                   moo_model_size='small')
    #
    # print('init core time: {}'.format(round(time.time() - t0, 4)))
    #
    # # %%e
    # x0 = problem_small.original_x
    #
    # t0 = time.time()
    # f0 = problem_small.evaluate(problem_small.original_x)
    # print('eval time: {:.4} s'.format(time.time() - t0))
    #
    # t0 = time.time()
    # dx = problem_small.gradient(problem_small.original_x)
    # print('grad time: {:.4} s'.format(time.time() - t0))
    #
    # t0 = time.time()
    # dx_1 = pinv(dx)
    # print('inverse time: {:.4} s'.format(time.time() - t0))
    #
    # # %%
    # print('small: {}, medium: {}'.format(f0, f0_medium))
