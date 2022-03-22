import os
import time
from pprint import pprint

import joblib
import numpy as np
import pandas as pd
from numpy.linalg import pinv
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination, get_reference_directions
from pymoo.optimize import minimize
from tabulate import tabulate
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer

from src.moo.core.continuation import BiDirectionalDsContinuation
from src.moo.factory import get_corrector, get_predictor, get_tfun, get_cont_termination
from src.moo.nn.problem import TsQuantileProblem, GradientTestsProblem
from src.moo.nn.utils import predict_from_batches, batch_array
from src.timeseries.utils.continuation import get_q_moo_params_for_problem
from src.timeseries.utils.filename import get_result_folder
from src.timeseries.utils.moo import get_hypervolume
from src.utils.plot import plot_bidir_2D_points_vectors, plot_2D_points_traces, plot_2D_points_vectors, plot_traces

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

    # outputs, output_map, data_map = model_params['model'].predict_all(valid, batch_size=128)

    # %%
    t0 = time.time()
    limits = np.array([1., 1.])
    problem = GradientTestsProblem(y_train=model_params['y_train'],
                                   x_train=model_params['x_train'],
                                   y_valid=model_params['y_valid'],
                                   x_valid=model_params['x_valid'],
                                   model=model_params['model'].model,
                                   eval_fs=[model_params['loss']],
                                   n_obj=2,
                                   constraints_limits=limits,
                                   quantile_ix=0,
                                   pred_batch_size=2 ** 9,
                                   grad_batch_size=2 ** 9,
                                   moo_model_size='small',
                                   grad_from_batches=False)

    print('init core time: {}'.format(round(time.time() - t0, 4)))

    # %%
    # import tensorflow as tf
    #
    # batch_size = 100000
    # self = problem
    # self.moo_model_input_train_batched = batch_array(self.moo_model_input_train, batch_size=batch_size)
    # self.y_train_batches = batch_array(self.y_train, batch_size=batch_size)
    #
    # individual = problem.original_x
    # if len(np.shape(individual)) == 1:
    #     individual = np.reshape(individual, (-1, 1))
    #
    # weights = self.individuals_to_weights(individual)
    # self.set_weights_moo_model(weights)
    #
    # batched_grads = []
    # for x, y in zip(self.moo_model_input_train_batched, self.y_train_batches):
    #     with tf.GradientTape(persistent=True) as tape:
    #         # grads are calculated always with train data
    #         y_pred = self.moo_model(x)
    #         loss = self.eval_fs[0](y, y_pred)
    #
    #     grads = [tape.gradient(loss[self.quantile_ix][i], self.moo_model.trainable_variables) for i in [0, 1]]
    #
    #     # numpy_grads = [[w.numpy() for w in d] for d in grads]
    #     # grads_reshaped = [self.weights_to_individuals(grad) for grad in numpy_grads]
    #     # batched_grads.append(np.vstack(grads_reshaped))
    #     weights_grads = [[w for w in d] for d in grads]
    #     grads_reshaped = [tf.concat([tf.reshape(w, [1, -1]) for w in grad], 1) for grad in weights_grads]
    #     batched_grads.append(tf.squeeze(tf.stack(grads_reshaped, axis=0)))
    #
    # # batched_grads = np.array(batched_grads)
    # # agg_grad = np.mean(batched_grads, axis=0)
    # batched_grads = tf.stack(batched_grads)
    # agg_grad = tf.reduce_mean(batched_grads, axis=0).numpy()
    #
    # with tf.GradientTape(persistent=True) as tape:
    #     # grads are calculated always with train data
    #     y_pred = self.moo_model(self.moo_model_input_train)
    #     loss = self.eval_fs[0](self.y_train, y_pred)
    #
    # grads = [tape.gradient(loss[self.quantile_ix][i], self.moo_model.trainable_variables) for i in [0, 1]]
    #
    # numpy_grads = [[w.numpy() for w in d] for d in grads]
    # grads_reshaped = [self.weights_to_individuals(grad) for grad in numpy_grads]
    # full_grad = np.vstack(grads_reshaped)
    #
    # diff = full_grad - agg_grad
    # print(np.sum(np.abs(diff)))

    # %%e
    t0 = time.time()
    f0 = problem.evaluate(problem.original_x)
    print('eval time: {:.4} s'.format(time.time() - t0))

    t0 = time.time()
    dx = problem.gradient(problem.original_x)
    print('grad time: {:.4} s'.format(time.time() - t0))

    t0 = time.time()
    dx_1 = pinv(dx)
    print('inverse time: {:.4} s'.format(time.time() - t0))

    # %%
    if solve_moea:
        pop, n_gen = 20, 20
        ref_dirs = get_reference_directions("energy", problem.n_obj, pop)
        algorithm = NSGA3(
            pop_size=pop,
            ref_dirs=ref_dirs,
            n_offsprings=pop,
            sampling=get_sampling("real_random"),
            crossover=get_crossover("real_sbx", prob=0.9, eta=15),
            mutation=get_mutation("real_pm", eta=20),
            eliminate_duplicates=True
        )

        termination = get_termination("n_gen", n_gen)

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

    batches_cfgs = [(2 ** i, True) for i in range(8, 16, 2)]
    batches_cfgs += [(len(problem.x_train), False)]

    cfgs_results = []
    for grad_batch_size, grad_from_batches in batches_cfgs:
        problem.grad_from_batches = grad_from_batches
        problem.grad_batch_size = grad_batch_size

        predictor = get_predictor('no_adjustment', problem=problem)

        corrector = get_corrector('delta_criteria',
                                  problem=problem,
                                  t_fun=get_tfun('weighted_dominance',
                                                 problem=problem,
                                                 eps=1e-4,
                                                 maxiter=50),
                                  a_fun=lambda a, dx: a,
                                  step_eps=2e-2,
                                  in_pf_eps=1e-4,
                                  maxiter=20)

        ds_cont = BiDirectionalDsContinuation(problem=problem,
                                              predictor=predictor,
                                              corrector=corrector,
                                              limits=limits,
                                              step_eps=2e-2,
                                              termination=get_cont_termination('tol', tol=8e-4),
                                              history=True)

        t0 = time.time()
        problem.n_f_evals, problem.n_grad_evals = 0, 0
        results = ds_cont.run(np.reshape(problem.original_x, (-1)))
        print('time: {} s'.format(round(time.time() - t0, 4)))
        print('f(x) evals: {}, dx(x) evals: {}'.format(problem.n_f_evals, problem.n_grad_evals))
        print(tabulate([[name, *inner.values()] for name, inner in results['evaluations'].items()],
                       headers=list(results['evaluations'][list(results['evaluations'].keys())[0]].keys()),
                       tablefmt='psql'))

        cfgs_results.append(results)

    hvs = {}
    for key, r in zip(batches_cfgs, cfgs_results):
        hvs[str(key)] = round(get_hypervolume(r['population']['F'], ref=[2., 2.]), 6)

    print(tabulate(hvs.items(),
                   headers=['batch cfg', 'hypervolume'],
                   tablefmt='psql',
                   floatfmt=(None, ",.6f"),
                   stralign="right"))

    # %%
    import pandas as pd
    from ast import literal_eval as make_tuple

    df = pd.DataFrame()
    df['hypervolume'] = hvs.values()
    df.index = hvs.keys()
    df['batch_size'] = [make_tuple(k)[0] for k in list(hvs.keys())]
    df['f_evals'] = [res['evaluations']['f']['total'] for res in cfgs_results]
    df['grad_evals'] = [res['evaluations']['grad']['total'] for res in cfgs_results]

    df.sort_values(by=['hypervolume'], ascending=False, inplace=True)
    print(tabulate(df, headers='keys', tablefmt='psql'))

    # import matplotlib.pyplot as plt
    # import seaborn as sns
    #
    # sns.set_theme('poster')
    # fig, ax = plt.subplots(figsize=(14, 8))
    # ax2 = plt.twinx()
    # sns.scatterplot(data=df, x='batch_size', y='hypervolume', color="g", ax=ax, label='hypervolume')
    # sns.scatterplot(data=df, x='batch_size', y='f_evals', ax=ax2, label='f_evals')
    # sns.scatterplot(data=df, x='batch_size', y='grad_evals', ax=ax2, label='grad_evals')
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    # ax2.legend(bbox_to_anchor=(1.05, .8), loc='upper left', borderaxespad=0)
    # plt.tight_layout()
    # plt.show()

    # %%
    file_path = os.path.join(results_folder, 'img', cfg['problem_name'] + '_batche_sizes')

    figs_data = []
    pareto = {'pf': None}
    for results in cfgs_results:
        plot_populations = [res['population'] for res in results['independent']]

        figs_data.append(plot_bidir_2D_points_vectors(plot_populations,
                                                      pareto,
                                                      arrow_scale=0.4,
                                                      markersize=5,
                                                      pareto_marker_mode='markers+lines',
                                                      save=False,
                                                      plot_arrows=True,
                                                      plot_points=True,
                                                      plot_ps=False,
                                                      return_fig=True).data)

    plot_traces(figs_data,
                save=cfg['save_plots'],
                file_path=file_path)

    points_traces = [r['population']['F'][np.argsort(r['population']['F'][:, 0])] for r in cfgs_results]

    plot_2D_points_traces(points_traces,
                          names=[str(b) for b in batches_cfgs],
                          markersizes=5,
                          color_ixs=None,
                          modes=['lines+markers'] * len(points_traces),
                          save=False,
                          save_png=False,
                          file_path=None,
                          title='HVs: ' + str(hvs),
                          size=(1980, 1080))
