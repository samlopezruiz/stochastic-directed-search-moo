import os
import time

from pymoo.algorithms.moo.nsga3 import NSGA3

from src.timeseries.moo.sds.config import sds_cfg
from src.timeseries.moo.core.harness import get_model_and_params, get_ts_problem, get_continuation_method, \
    run_cont_problem, save_cont_resuls, save_latex_table, plot_pf_and_total, filename_from_cfg, plot_2D_pf
from src.timeseries.moo.sds.utils.bash import get_input_args
from src.timeseries.moo.sds.utils.indicators import metrics_of_pf
from src.timeseries.moo.sds.utils.util import get_from_dict, set_in_dict
from src.timeseries.utils.files import save_vars
from src.timeseries.utils.moo import sort_1st_col
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination, get_reference_directions
from pymoo.optimize import minimize

if __name__ == '__main__':
    # %%
    input_args = get_input_args()

    cfg = {'save_plots': False,
           'save_results': False,
           'save_latex': False,
           'plot_title': False,
           }

    project = 'snp'

    sds_cfg['model']['ix'] = input_args['model_ix']
    sds_cfg['model']['ix'] = 'standalone'

    if sds_cfg["problem"]["split_model"] == 'small':
        # Limit PF only for small size problem, since MOEAs cannot
        # find a limited PF for the medium size problem
        set_in_dict(sds_cfg, ['sds', 'max_increment'], 0.05)

    set_in_dict(sds_cfg, ['sds', 'step_size'], 5e-3)

    model_params, results_folder = get_model_and_params(sds_cfg, project)
    problem = get_ts_problem(sds_cfg, model_params, test_ss=False)
    ds_cont = get_continuation_method(sds_cfg, problem)

    # %% Optimize with SDS
    results, metrics = run_cont_problem(ds_cont, problem)
    # Save results
    save_cont_resuls({'results': results, 'metrics': metrics, 'cont_cfg': sds_cfg}, results_folder, cfg, sds_cfg)
    # Save latex tables
    save_latex_table(metrics, results_folder, cfg, sds_cfg)
    # Plot results
    plot_pf_and_total(results, results_folder, cfg, sds_cfg)

    # %% Optimize with MOEA
    problem.n_constr = 2
    if sds_cfg["problem"]["split_model"] == 'small':
        problem.constraints_limits = [0.459, .583]
        pop_size, n_gen = 78, 20 #400
    else:
        problem.constraints_limits = [1.0, 1.0]
        pop_size, n_gen = 60, 150

    t0 = time.time()
    algorithm = NSGA2(
        pop_size=pop_size,
        n_offsprings=pop_size,
        ref_dirs=get_reference_directions("energy", problem.n_obj, pop_size),
        sampling=get_sampling("real_random"),
        crossover=get_crossover("real_sbx", prob=0.9, eta=15),
        mutation=get_mutation("real_pm", eta=20),
        eliminate_duplicates=True
    )

    termination = get_termination("n_gen", n_gen)

    nsga2_res = minimize(problem,
                         algorithm,
                         termination,
                         seed=42,
                         save_history=False,
                         verbose=True)

    # %%
    t0 = time.time()
    algorithm = NSGA3(
        pop_size=pop_size,
        n_offsprings=pop_size,
        ref_dirs=get_reference_directions("energy", problem.n_obj, pop_size),
        sampling=get_sampling("real_random"),
        crossover=get_crossover("real_sbx", prob=0.9, eta=15),
        mutation=get_mutation("real_pm", eta=20),
        eliminate_duplicates=True
    )

    termination = get_termination("n_gen", n_gen)

    nsga3_res = minimize(problem,
                         algorithm,
                         termination,
                         seed=42,
                         save_history=False,
                         verbose=True)

    # %%
    X_nsga2_sorted, F_nsga2_sorted = sort_1st_col(nsga2_res.X, nsga2_res.F)
    X_nsga3_sorted, F_nsga3_sorted = sort_1st_col(nsga3_res.X, nsga3_res.F)
    X_sorted, F_sorted = sort_1st_col(results['population']['X'], results['population']['F'])
    fx_ini = results['independent'][0]['descent']['ini_fx'].reshape(1, 2)
    filename = filename_from_cfg(sds_cfg)

    plot_2D_pf(Fs=[F_sorted, F_nsga2_sorted, F_nsga3_sorted],
               fx_inis=[fx_ini, fx_ini, fx_ini],
               names=['SDS', 'NSGA-II', 'NSGA-III', 'ini', 'ini', 'ini'],
               f_markersize=6,
               colors_ixs=[0, 2, 1, 10, 10, 10],
               save=cfg['save_plots'],
               label_scale=1.7,
               size=(1000, 700),
               save_pdf=True,
               img_path=os.path.join(results_folder, 'sds', 'img', filename))

    #%%
    save_vars({'sds': {'F': F_sorted, 'X': X_sorted, 'results': results},
               'nsga2': {'F': F_nsga2_sorted, 'X': X_nsga2_sorted},
               'nsga3': {'F': F_nsga3_sorted, 'X': X_nsga3_sorted}},
              os.path.join(results_folder, 'comparison', filename))
