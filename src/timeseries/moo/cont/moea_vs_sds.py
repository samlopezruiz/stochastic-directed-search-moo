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
#%%
    sds_results, sds_metrics = [], []
    n_repeat = 10
    seeds = range(n_repeat)
    for i, seed in enumerate(seeds):
        sds_metrics.append(joblib.load(f'tmp/sds_results{seed}.pkl'))

    #%%
    rhos = np.array([r['times'].loc['J(x)', 'mean (s)'] / r['times'].loc['f(x)', 'mean (s)']  for r in sds_metrics])
    fevals = np.array([r['pred_corr_metrics']['f_evals'].sum() for r in sds_metrics])
    Jevals = np.array([r['pred_corr_metrics']['grad_evals'].sum() for r in sds_metrics])
    evals = Jevals * rhos + fevals
    print(np.mean(evals), np.std(evals))

    times = [r['times'].loc['execution', 'mean (s)'] for r in sds_metrics]
    print(np.mean(times), np.std(times))

    print(np.mean([r['subset_metrics'].loc['valid', 'mean norm'] for r in sds_metrics]), np.std([r['subset_metrics'].loc['valid', 'std norm'] for r in sds_metrics]))

    sds_hvs = [r['subset_metrics'].loc['valid', 'hv'] for r in sds_metrics]
    print(np.mean(sds_hvs), np.std(sds_hvs))

    # %%
    moea_results = []
    times = []
    for seed in range(n_repeat):
        moea_metrics, t = joblib.load(f'tmp/moea_results_{seed}.pkl')
        moea_results.append(moea_metrics)
        times.append(t)

    #%%
    gens_to_measure = [14, 25, 50, 75, 100, 125, 150]
    for c, g in enumerate(gens_to_measure):
        print('-' * 20)
        print(f'generation={g}')
        ts = [t[c] for t in times]
        print(np.mean(ts), np.std(ts))

        distances = [r[c]['distances'] for r in moea_results]
        distances = [item for listoflists in distances for item in listoflists]
        print(np.mean(distances), np.std(distances))

        hvs = [r[c]['hv'] for r in moea_results]
        print(np.mean(hvs), np.std(hvs))

        # ws = wilcoxon_significance([sds_hvs, hvs], ['SDS', 'NSGA-II'])
        # print('-'*20)
        # print(ws)

    # %%
    # results = sds_results[0]
    # X_moea_sorted, F_moea_sorted = sort_1st_col(res.X, res.F)
    # X_sorted, F_sorted = sort_1st_col(results['population']['X'], results['population']['F'])
    # fx_ini = results['independent'][0]['descent']['ini_fx'].reshape(1, 2)
    # filename = filename_from_cfg(cont_cfg)
    #
    # plot_2D_pf(Fs=[F_sorted, F_moea_sorted],
    #        fx_inis=[fx_ini, fx_ini],
    #        names=['SDS', 'NSGA-II', 'ini', 'ini'],
    #        f_markersize=6,
    #        colors_ixs=[0, 2, 10, 10],
    #        save=False,
    #        label_scale=1.8)