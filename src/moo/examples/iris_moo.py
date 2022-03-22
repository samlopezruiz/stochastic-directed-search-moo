import numpy as np
import plotly.io as pio
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.factory import get_termination
from pymoo.optimize import minimize

from src.moo.examples.nn_standalone.iris import iris_model, iris_data
from src.moo.examples.utils.metrics import classification_eval_metrics

from src.moo.loss.classification import moo_classification_weights, WeightedCategoricalCrossentropy
from src.moo.nn.problem import ClassificationProblem
from src.utils.plot import plot_2D_points_traces

pio.renderers.default = "browser"
if __name__ == '__main__':
    # %%
    cfg = {'save_plots': False,
           'solve_moea': True,
           'plot_boxes': False
           }

    X_train, X_test, y_train, y_test = iris_data()
    model = iris_model(weights=np.ones(3))
    model.fit(X_train, y_train, epochs=100, verbose=0)

    recall = classification_eval_metrics(X_train, y_train, model)['recall']
    print('recall: {}'.format(recall))

    moo_weights = moo_classification_weights(n_classes=3, individual_weight=2.)
    # loss_funcs = [WeightedCategoricalCrossentropy(w, from_logits=False, fp_mode=True).loss for w in moo_weights]
    loss_funcs = [WeightedCategoricalCrossentropy(from_logits=False, fp_mode=mode).loss for mode in [True, False]]
    problem = ClassificationProblem(y_train,
                                    X_train,
                                    model,
                                    eval_fs=loss_funcs)

    # %%
    if cfg['solve_moea']:
        algorithm = NSGA2(
            pop_size=100,
            n_offsprings=100,
            sampling=get_sampling("real_random"),
            crossover=get_crossover("real_sbx", prob=0.9, eta=15),
            mutation=get_mutation("real_pm", eta=20),
            eliminate_duplicates=True,
        )

        termination = get_termination("n_gen", 100)

        res = minimize(problem,
                       algorithm,
                       termination,
                       seed=1,
                       save_history=False,
                       verbose=True)

        X = res.X
        F = res.F

        ixs = np.argsort(F[:, 0])
        pf_moea = F[ixs]
        ps_moea = X[ixs]

        plot_2D_points_traces([pf_moea])

        # plot_points_3d(pf_moea, markersize=7, title='MOEA method Pareto Front')

        precision = problem.eval_with_metrics(ps_moea, key='precision')
        recall = problem.eval_with_metrics(ps_moea, key='recall')
        orig_metrics = problem.eval_with_metrics(problem.original_x)
        plot_2D_points_traces([precision, recall])
        # plot_points_3d(precision, markersize=7, title='MOEA precision')
        # plot_points_3d(precision, markersize=7, title='MOEA precision')

    #%%

    # %%
    # x0 = None
    # f0 = problem.evaluate(x0)
    #
    # f_limits = np.array([[0., 1.]] * 3)
    #
    # predictor = get_predictor('no_adjustment',
    #                           problem=problem,
    #                           eps=1e-2)
    #
    # corrector = get_corrector('delta_criteria',
    #                           problem=problem,
    #                           t_fun=get_tfun('weighted_dominance',
    #                                          problem=problem,
    #                                          eps=1e-5,
    #                                          maxiter=50),
    #                           a_fun=lambda a, dx: a,
    #                           step_eps=9e-2,
    #                           in_pf_eps=1e-2,
    #                           maxiter=100)
    #
    # ds_cont = ContinuationBoxes(problem=problem,
    #                             predictor=predictor,
    #                             corrector=corrector,
    #                             f_limits=f_limits,
    #                             c=0.8,
    #                             # h_max=14,
    #                             step_eps=9e-2,
    #                             debug=True,
    #                             history=True,
    #                             )
    #
    # t0 = time.time()
    # problem.n_f_evals, problem.n_grad_evals = 0, 0
    # results = ds_cont.run(x0)
    # print('time: {} s'.format(round(time.time() - t0, 4)))
    # print('f(x) evals: {}, dx(x) evals: {}'.format(problem.n_f_evals, problem.n_grad_evals))
    # pprint(results['evaluations'])

    # %% Hypervolume
    # points = ds_cont.boxes.get_points()
    # print('{} solutions found, {} best solutions found'.format(len(points['fx']),
    #                                                            len(points['fx'][points['best_ix'], :])))
    # hv_moea = hypervolume(pf_moea, ref=[2., 2., 2.]) if solve_moea else np.nan
    # hv_pop = hypervolume(results['population']['F'], ref=[2., 2., 2.])
    # hv_ref = hypervolume(testfun['pf'], ref=[2., 2., 2.])
    # print('ref hv: {}, pop hv: {}, moea hv:{}'.format(hv_ref, hv_pop, hv_moea))

    # %%
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
