import os
import time

import joblib
from tabulate import tabulate

from src.models.compare.winners import wilcoxon_significance
from src.timeseries.moo.cont.core.config import cont_cfg
from src.timeseries.moo.cont.core.harness import get_model_and_params, get_ts_problem, get_continuation_method, \
    run_cont_problem, plot_pf_and_total, save_cont_resuls, save_latex_table, plot_2D_pf, filename_from_cfg
from src.timeseries.moo.cont.utils.bash import get_input_args
from src.timeseries.moo.cont.utils.indicators import metrics_of_pf
from src.timeseries.moo.cont.utils.util import set_in_dict, get_from_dict
from src.timeseries.utils.moo import sort_1st_col
from src.utils.plot import plot_2D_points_traces_total
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination, get_reference_directions
from pymoo.optimize import minimize
import numpy as np

if __name__ == '__main__':
    # %%
    input_args = get_input_args()

    cfg = {'save_plots': False,
           'save_results': False,
           'save_latex': False,
           'plot_title': False,
           }

    project = 'snp'

    cont_cfg['model']['ix'] = input_args['model_ix']
    cont_cfg['model']['ix'] = 5
    # cont_cfg['problem']['split_model'] = 'medium'
    print('Model ix: {}'.format(get_from_dict(cont_cfg, ['model', 'ix'])))
    # set_in_dict(cont_cfg, ['cont', 'max_increment'], 0.05)
    # set_in_dict(cont_cfg, ['cont', 'step_eps'], 5e-3)

    model_params, results_folder = get_model_and_params(cont_cfg, project)
    problem = get_ts_problem(cont_cfg, model_params, test_ss=False)
    ds_cont = get_continuation_method(cont_cfg, problem)
    results, metrics = run_cont_problem(ds_cont, problem)

    # #%%
    # # Save results
    # save_cont_resuls({'results': results, 'metrics': metrics, 'cont_cfg': cont_cfg}, results_folder, cfg, cont_cfg)
    #
    # # Save latex tables
    # save_latex_table(metrics, results_folder, cfg, cont_cfg)
    #
    # # Plot results
    # plot_pf_and_total(results, results_folder, cfg, cont_cfg)

    # %%
    # problem.constraints_limits = [0.459, .583]
    # # problem.constraints_limits = [1.0, 1.0]
    # problem.n_constr = 2
    # pop_size, n_gen = 78, 200
    #
    #
    # t0 = time.time()
    # algorithm = NSGA2(
    #     pop_size=pop_size,
    #     n_offsprings=pop_size,
    #     ref_dirs=get_reference_directions("energy", problem.n_obj, pop_size),
    #     sampling=get_sampling("real_random"),
    #     crossover=get_crossover("real_sbx", prob=0.9, eta=15),
    #     mutation=get_mutation("real_pm", eta=20),
    #     eliminate_duplicates=True
    # )
    #
    # termination = get_termination("n_gen", n_gen)
    #
    # res = minimize(problem,
    #                algorithm,
    #                termination,
    #                seed=42,
    #                save_history=False,
    #                verbose=True)
    #
    # F = problem.eval_individuals(res.X, 'valid')
    # X_moea_sorted, F_moea_sorted = sort_1st_col(res.X, F)
    # moea_metrics = metrics_of_pf(F_moea_sorted, ref=[2., 2.])
    #
    # #%%
    # X_moea_sorted, F_moea_sorted = sort_1st_col(res.X, res.F)
    # X_sorted, F_sorted = sort_1st_col(results['population']['X'], results['population']['F'])
    # fx_ini = results['independent'][0]['descent']['ini_fx'].reshape(1, 2)
    # filename = filename_from_cfg(cont_cfg)
    #
    # plot_2D_pf(Fs=[F_sorted, F_moea_sorted],
    #            fx_inis=[fx_ini, fx_ini],
    #            names=['SDS', 'NSGA-II', 'ini', 'ini'],
    #            f_markersize=8,
    #            colors_ixs=[0, 2, 10, 10],
    #            save=False,
    #            label_scale=1.8,
    #            img_path=os.path.join(results_folder, 'cont', 'img', filename))

    # %%
    # X_sorted, F_sorted = sort_1st_col(results['population']['X'], results['population']['F'])
    # fx_ini = results['independent'][0]['descent']['ini_fx'].reshape(1, 2)

    # plot_2D_points_traces_total([F_sorted, fx_ini],
    #                             names=['cont', 'ini'],
    #                             color_ixs=[0, 10],
    #                             markersizes=[10, 20],
    #                             marker_symbols=['circle', 'star'],
    #                             modes=['markers+lines', 'markers'],
    #                             outlines=[False, True],
    #                             axes_labels=('Objective 1', 'Objective 2'),
    #                             label_scale=1.8)
    #
    # # %% Print metrics
    # print(tabulate(metrics['pred_corr_metrics'], headers='keys', tablefmt='psql'))
    # print(tabulate(metrics['subset_metrics'], headers='keys', tablefmt='psql'))
    # print(tabulate(metrics['times'], headers='keys', floatfmt=(None, ",.4f"), tablefmt='psql'))
