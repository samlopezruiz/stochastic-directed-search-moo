import os
import time
from pprint import pprint

import numpy as np
import plotly.io as pio
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.factory import get_termination
from pymoo.optimize import minimize

from src.moo.examples.nn_standalone.fashion_mnist import mnist_fashion_model, fashion_mnist_data
from src.moo.examples.utils.metrics import classification_eval_metrics

from src.moo.core.continuation import BiDirectionalDsContinuation
from src.moo.factory import get_tfun, get_corrector, get_predictor, get_cont_termination
from src.moo.loss.classification import WeightedCategoricalCrossentropy
from src.moo.nn.problem_old import ClassificationProblem
from src.moo.utils.functions import sort_according_first_obj
from src.utils.plot import plot_2D_points_traces, plot_metrics_traces, \
    plot_bidir_2D_points_vectors

pio.renderers.default = "browser"
if __name__ == '__main__':
    # %%
    cfg = {'save_plots': False,
           'solve_moea': False,
           'plot_boxes': False
           }

    n_classes = 10
    X_train, X_test, y_train, y_test, class_names = fashion_mnist_data()
    model = mnist_fashion_model(weights=np.ones(n_classes))
    model.fit(X_train, y_train, epochs=30, verbose=1)

    recall = classification_eval_metrics(X_train, y_train, model)['recall']
    print('recall: {}'.format(recall))

    # moo_weights = moo_classification_weights(n_classes=n_classes, individual_weight=2.)
    loss_funcs = [WeightedCategoricalCrossentropy(from_logits=True, fp_mode=mode).loss for mode in [True, False]]
    # loss_funcs = [WeightedCategoricalCrossentropy(w, from_logits=True).loss for w in moo_weights]
    problem = ClassificationProblem(y_train,
                                    X_train,
                                    model,
                                    eval_fs=loss_funcs)

    x0 = problem.original_x
    f0 = problem.evaluate(x0)

    # %%
    if cfg['solve_moea']:
        algorithm = NSGA2(
            pop_size=200,
            n_offsprings=200,
            sampling=get_sampling("real_random"),
            crossover=get_crossover("real_sbx", prob=0.9, eta=15),
            mutation=get_mutation("real_pm", eta=20),
            eliminate_duplicates=True,
        )

        termination = get_termination("n_gen", 200)

        res = minimize(problem,
                       algorithm,
                       termination,
                       seed=1,
                       save_history=False,
                       verbose=True)

        pf_moea, ps_moea = sort_according_first_obj(res.F, res.X)
        plot_2D_points_traces([pf_moea, f0])

        precision = problem.eval_with_metrics(ps_moea, key='precision')
        recall = problem.eval_with_metrics(ps_moea, key='recall')
        plot_metrics_traces(traces=[precision, recall], subtitles=['precision', 'recall'], x_labels=None)

        metrics = np.vstack([np.mean(precision, axis=1), np.mean(recall, axis=1)])
        orig_metric = np.array([problem.eval_with_metrics(problem.original_x, key='precision'),
                                problem.eval_with_metrics(problem.original_x, key='recall')])
        plot_2D_points_traces([metrics, orig_metric], names=['precision', 'recall'])

    # %%
    # x0 = None
    # f0 = problem.evaluate(x0)

    f_limits = np.array([[0., 1.]] * 2)

    predictor = get_predictor('no_adjustment',
                              problem=problem,
                              eps=1e-2)

    corrector = get_corrector('delta_criteria',
                              problem=problem,
                              t_fun=get_tfun('weighted_dominance',
                                             problem=problem,
                                             eps=1e-5,
                                             maxiter=50),
                              a_fun=lambda a, dx: a,
                              step_eps=3e-2,
                              in_pf_eps=1e-2,
                              maxiter=100)

    ds_cont = BiDirectionalDsContinuation(problem=problem,
                                          predictor=predictor,
                                          corrector=corrector,
                                          termination=get_cont_termination('tol', tol=4e-4),
                                          # f_limits=f_limits,
                                          step_eps=9e-2)

    t0 = time.time()
    problem.n_f_evals, problem.n_grad_evals = 0, 0
    results = ds_cont.run(x0)
    print('time: {} s'.format(round(time.time() - t0, 4)))
    print('f(x) evals: {}, dx(x) evals: {}'.format(problem.n_f_evals, problem.n_grad_evals))
    pprint(results['evaluations'])

    # %%
    # pf_cont, ps_cont = sort_according_first_obj(results['population']['F'], results['population']['X'])
    # metrics = np.vstack([problem.eval_with_metrics(ps_cont, key='precision'),
    #                      problem.eval_with_metrics(ps_cont, key='recall')])
    # orig_metrics = np.array([problem.eval_with_metrics(problem.original_x, key='precision'),
    #                          problem.eval_with_metrics(problem.original_x, key='recall')])
    #
    # plot_2D_points_traces([metrics, orig_metrics], names=['population', 'original'])

    file_path = os.path.join('../img', cfg['problem_name'])
    plot_populations = [res['population'] for res in results['independent']]
    plot_bidir_2D_points_vectors(plot_populations,
                                 pareto=None,
                                 arrow_scale=0.4,
                                 markersize=5,
                                 save=cfg['save_plots'],
                                 save_png=False,
                                 file_path=file_path,
                                 size=(1980, 1080),
                                 plot_arrows=True,
                                 plot_points=True)

    # %% Hypervolume
    # points = ds_cont.boxes.get_points()
    # print('{} solutions found, {} best solutions found'.format(len(points['fx']),
    #                                                            len(points['fx'][points['best_ix'], :])))
    # hv_moea = hypervolume(pf_moea, ref=[2., 2., 2.]) if solve_moea else np.nan
    # hv_pop = hypervolume(results['population']['F'], ref=[2., 2., 2.])
    # hv_ref = hypervolume(testfun['pf'], ref=[2., 2., 2.])
    # print('ref hv: {}, pop hv: {}, moea hv:{}'.format(hv_ref, hv_pop, hv_moea))
    #
    # # %%
    # file_path = os.path.join('../img')
    #
    # if plot_boxes:
    #     boxes_edges = ds_cont.boxes.get_boxes()
    #     box_fig = plot_boxes_3d(boxes_edges, return_fig=True)
    #
    #     pts_fig = plot_points_3d(points['fx'],
    #                              secondary=points['c'],
    #                              mask=points['best_ix'],
    #                              return_fig=True,
    #                              markersize=5)
    #
    #     plot_traces([box_fig.data, pts_fig.data])
    #
    # plot_points_3d(points['fx'],
    #                secondary=points['c'],
    #                mask=points['best_ix'],
    #                markersize=5,
    #                only_best=False,
    #                title='Continuation method Pareto Front')
    #
    # if solve_moea:
    #     plot_points_3d(pf_moea, markersize=7, title='MOEA method Pareto Front')
    #
    # # %%
    # F = points['fx']
    # ref_dirs = get_reference_directions("energy", problem.n_obj, 500)
    # pop = pop_from_array_or_individual(F)
    # pop.set('F', F)
    # rds = ReferenceDirectionSurvival(ref_dirs)
    # niching = rds.do(problem, pop, n_survive=500)
    # opt = rds.opt.get('F')
    # plot_points_3d(opt,
    #                markersize=8,
    #                title='Continuation method Pareto Front with nitching')
    # %%
