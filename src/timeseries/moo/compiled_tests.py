import gc
import math
import os
import time
from copy import deepcopy

import tensorflow as tf
from pymoo.algorithms.moo.nsga3 import NSGA3

from src.timeseries.moo.core.harness import get_model_and_params, get_ts_problem, get_continuation_method, \
    run_cont_problem, save_cont_resuls, save_latex_table, plot_pf_and_total, filename_from_cfg, plot_2D_pf
from src.timeseries.moo.sds.utils.util import get_from_dict, set_in_dict
from src.timeseries.utils.files import save_vars
from src.timeseries.utils.moo import sort_1st_col
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination, get_reference_directions
from pymoo.optimize import minimize
from src.timeseries.moo.sds.config import sds_cfg, params_cfg, experiments_cfg
from src.timeseries.moo.core.harness import get_cfg_from_loop_cfg, \
    get_sublevel_keys, run_experiments


import itertools
import os

import joblib
import numpy as np
import pandas as pd
import plotly.io as pio
from plotly.subplots import make_subplots
from tabulate import tabulate

from src.models.compare.winners import wilcoxon_rank, wilcoxon_significance, kruskal_significance
from src.timeseries.moo.core.harness import plot_2D_pf
from src.timeseries.moo.sds.utils.results import get_compiled_dict, df_from_dict, compile_metrics, combine_means_stds, \
    adapt_runs
from src.timeseries.moo.sds.utils.util import get_from_dict
from src.timeseries.utils.filename import get_result_folder
from src.timeseries.utils.moo import sort_arr_1st_col
from src.timeseries.utils.util import concat_mean_std_df, write_text_file, latex_table, mean_std_from_array
from src.utils.plot import plot_bidir_2D_points_vectors, \
    plot_pfs, bar_plots_with_errors, bar_plot_3axes_with_errors, plot_2D_points_traces, plot_boxes, \
    plot_2d_grouped_traces, marker_names, color_sequences, plot_radar, set_fig_font_scale, box_plot_colors


def run_sds_vs_moea_optimization(project, cfg, run_moeas):
    sds_cfg_ = deepcopy(sds_cfg)
    sds_cfg_['model']['ix'] = 'standalone'

    if sds_cfg_["problem"]["split_model"] == 'small':
        # Limit PF only for small size problem, since MOEAs cannot
        # find a limited PF for the medium size problem
        set_in_dict(sds_cfg_, ['sds', 'max_increment'], 0.05)

    set_in_dict(sds_cfg_, ['sds', 'step_size'], 5e-3)

    model_params, results_folder = get_model_and_params(sds_cfg_, project)
    problem = get_ts_problem(sds_cfg_, model_params, test_ss=False)
    ds_cont = get_continuation_method(sds_cfg_, problem)

    # %% Optimize with SDS
    results, metrics = run_cont_problem(ds_cont, problem)
    # Save results
    save_cont_resuls({'results': results, 'metrics': metrics, 'cont_cfg': sds_cfg_}, results_folder, cfg, sds_cfg_)
    # Save latex tables
    save_latex_table(metrics, results_folder, cfg, sds_cfg_)
    # Plot results
    plot_pf_and_total(results, results_folder, cfg, sds_cfg_)

    if run_moeas:
        # %% Optimize with MOEA
        problem.n_constr = 2
        if sds_cfg_["problem"]["split_model"] == 'small':
            problem.constraints_limits = [0.459, .583]
            pop_size, n_gen = 78, 400
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

        # %% Execute NGSA-II
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

        # %% Execute NSGA-III
        X_nsga2_sorted, F_nsga2_sorted = sort_1st_col(nsga2_res.X, nsga2_res.F)
        X_nsga3_sorted, F_nsga3_sorted = sort_1st_col(nsga3_res.X, nsga3_res.F)
        X_sorted, F_sorted = sort_1st_col(results['population']['X'], results['population']['F'])
        fx_ini = results['independent'][0]['descent']['ini_fx'].reshape(1, 2)
        filename = filename_from_cfg(sds_cfg_)

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

        save_vars({'sds': {'F': F_sorted, 'X': X_sorted, 'results': results},
                   'nsga2': {'F': F_nsga2_sorted, 'X': X_nsga2_sorted},
                   'nsga3': {'F': F_nsga3_sorted, 'X': X_nsga3_sorted}},
                  os.path.join(results_folder, 'comparison', filename))

    tf.keras.backend.clear_session()
    gc.collect()

def run_sds_experiment(project, cfg, experiment, n_runs=None):
    model = 'standalone'

    exp_cfg = experiments_cfg[experiment]

    if n_runs is not None:
        exp_cfg['ini_cfg']['seeds'] = np.arange(0, n_runs)

    set_in_dict(sds_cfg, ['model', 'ix'], model)
    set_in_dict(sds_cfg, ['sds', 'verbose'], False)

    print('Model ix: {}'.format(sds_cfg['model']['ix']))

    relevant_cfg = [params_cfg[k] for k in get_sublevel_keys(exp_cfg['loop_cfg'], [])]
    cfg['experiment_name'] = '_'.join([c['keys'][-1] for c in relevant_cfg])
    cfgs = get_cfg_from_loop_cfg(exp_cfg['loop_cfg'], params_cfg, sds_cfg, [])
    experiment_labels = [dict([(c['keys'][-1], get_from_dict(cfg, c['keys'])) for c in relevant_cfg]) for cfg in cfgs]

    print('-----EXPERIMENTS-----')
    header = experiment_labels[0].keys()
    rows = [x.values() for x in experiment_labels]
    print(tabulate(rows, header, tablefmt='psql'))

    # %% Run experiments
    exp_results, results_folder = run_experiments(cfgs,
                                                  project,
                                                  relevant_cfg,
                                                  get_model=exp_cfg['ini_cfg']['get_model'],
                                                  get_problem=exp_cfg['ini_cfg']['get_problem'],
                                                  get_cont=exp_cfg['ini_cfg']['get_cont'],
                                                  change_batch_size=exp_cfg['ini_cfg']['change_batch_size'],
                                                  use_gpu=exp_cfg['ini_cfg'].get('use_gpu', True),
                                                  seeds=exp_cfg['ini_cfg'].get('seeds', None))

    # %% Save results
    model_ix = get_from_dict(sds_cfg, ['model', 'ix'])
    filename = '{}_ix_{}_it'.format(model_ix, len(cfgs))
    save_vars({'params_cfg': params_cfg,
               'exp_cfg': exp_cfg,
               'exp_lbl': experiment_labels,
               'exp_results': exp_results},
              os.path.join(os.path.dirname(results_folder),
                           'experiments',
                           cfg['experiment_name'],
                           filename),
              )

    tf.keras.backend.clear_session()
    gc.collect()


def explore_experiment_results(project, cfg, experiment, experiment_results_cfg, hv_min, hv_max):
    cfg['experiment_name'] = experiment

    def pad_str_num_cols(df, round_dig=3, thold=1e-1, inplace=True):
        if not inplace:
            df = df.copy()

        for col in df.columns:
            if max(df[col]) < thold:
                df[col] = df[col].apply(lambda x: "{:.2e}".format(x))
            else:
                if round_dig == 0:
                    digits = df[col].astype(float).round(round_dig).astype(int).astype(str)
                    df[col] = digits.str.rjust(digits.str.len().max(), fillchar=' ')
                else:
                    digits = df[col].astype(float).round(round_dig).astype(str).str.split('.')
                    df_dig = pd.DataFrame(digits.to_list(), columns=['i', 'd'], index=digits.index)
                    df_dig['i'] = df_dig['i'].str.rjust(df_dig['i'].str.len().max(), fillchar=' ')
                    df_dig['d'] = df_dig['d'].str.ljust(df_dig['d'].str.len().max(), fillchar='0')
                    df[col] = df_dig['i'] + '.' + df_dig['d']

        if not inplace:
            return df

    def save_df_to_latex(df, title, path):
        write_text_file(path, latex_table(title, df.to_latex(escape=False, column_format='r' + 'r' * df.shape[1])))

    file_cfg = [c for c in experiment_results_cfg if c['key'] == experiment][0]
    # base_path = os.path.join(get_result_folder({}, project), 'experiments', file_cfg['folder'])
    results_folder = os.path.join(get_result_folder({}, project), 'experiments', file_cfg['folder'])
    img_folder = os.path.join(results_folder, 'img')
    cont_results = joblib.load(
        os.path.join(get_result_folder({}, project), 'experiments', file_cfg['folder'], file_cfg['experiment']) + '.z')
    lbls = [', '.join(['{}:{}'.format(k, v) for k, v in d.items()]) for d in cont_results['exp_lbl']]

    if isinstance(file_cfg['prefix'], list):
        for p in file_cfg['prefix']:
            lbls = [s.replace(p[0], p[1]) if isinstance(p, tuple) else s.replace(p, '') for s in lbls]
    else:
        lbls = [s.replace(file_cfg['prefix'], '') for s in lbls]

    x_title = file_cfg['x_title']
    performance_title = file_cfg['x_title']

    print(lbls)

    # %% Use only one experiment file
    if isinstance(cont_results['exp_results'][0]['results'], list):
        subset_metrics_dfs = [[r['subset_metrics'] for r in res['metrics']] for res in cont_results['exp_results']]
        pred_corr_metrics_dfs = [[r['pred_corr_metrics'] for r in res['metrics']] for res in
                                 cont_results['exp_results']]
        times_dfs = [[r['times'] for r in res['metrics']] for res in cont_results['exp_results']]
    else:
        subset_metrics_dfs = [get_from_dict(res, ['metrics', 'subset_metrics']) for res in cont_results['exp_results']]
        pred_corr_metrics_dfs = [get_from_dict(res, ['metrics', 'pred_corr_metrics']) for res in
                                 cont_results['exp_results']]
        times_dfs = [get_from_dict(res, ['metrics', 'times']) for res in cont_results['exp_results']]

        # %%
    items_cfg = {
        'pred_norm': ['pred_corr_metrics', 'pred_norm'],
        'corr_norm': ['pred_corr_metrics', 'corr_norm'],
        'descent_norm': ['pred_corr_metrics', 'descent_norm'],
        'corrector steps': ['pred_corr_metrics', 'n_correctors'],
        'train hv': ['subset_metrics', 'train', 'hv'],
        'valid hv': ['subset_metrics', 'valid', 'hv'],
        'train dist': ['subset_metrics', 'train', 'distances'],
        'valid dist': ['subset_metrics', 'valid', 'distances'],
        'predictor f evals': ['evaluations', 'f', 'predictor'],
        'corrector f evals': ['evaluations', 'f', 'corrector'],
        'descent f evals': ['evaluations', 'f', 'descent'],
        'predictor g evals': ['evaluations', 'grad', 'predictor'],
        'corrector g evals': ['evaluations', 'grad', 'corrector'],
        'descent g evals': ['evaluations', 'grad', 'descent'],
    }

    results_items = {}
    for k, v in items_cfg.items():
        results_items[k] = [[get_from_dict(r, v) for r in res['results']] for res in cont_results['exp_results']]

    results_items['exec time'] = [[r['exec_time'] for r in res['results']] for res in cont_results['exp_results']]
    results_items['descent steps'] = [[len(l) for l in res] for res in results_items['descent_norm']]
    # merge experiment runs in a list of all runs per experiment
    flatten_exp_results = dict([(k, [(list(itertools.chain(*exp)) if isinstance(exp[0], list) else exp) for exp in v]) \
                                for k, v in results_items.items()])

    # %%
    plot_cfgs = [{'keys': ['pred_norm', 'corr_norm', ], 'color_position': 'auto',
                  'secondary_y': True, 'color_ixs': [0, 1, 3], 'name': 'norms'},
                 {'keys': ['train hv', 'valid hv'], 'color_position': 'auto',
                  'secondary_y': False, 'boxmode': 'overlay', 'y_title': 'hypervolume', 'name': 'hvs'},
                 {'keys': ['train dist', 'valid dist'], 'color_position': 'max',
                  'secondary_y': False, 'y_title': 'distance', 'name': 'dist'},
                 {'keys': ['corrector steps', 'descent steps'], 'color_position': 'auto',
                  'secondary_y': False, 'y_title': 'no. steps', 'name': 'n_steps'},

                 {'keys': ['predictor f evals', 'corrector f evals', 'descent f evals', 'predictor g evals',
                           'corrector g evals', 'descent g evals'],
                  'secondary_y': False, 'type': 'bar', 'name': 'other'},
                 ]

    wilcoxon_matrix = {}
    for plot_cfg in plot_cfgs:
        metrics = dict([(k, flatten_exp_results[k]) for k in plot_cfg['keys']])
        plot_cfg['wilcoxon'], plot_cfg['color_text'] = {}, {}
        plot_cfg['color'] = {}
        plot_cfg['metrics'] = metrics
        for key, list_of_lists in metrics.items():
            plot_cfg['wilcoxon'][key] = wilcoxon_significance(list_of_lists, labels=lbls, p_thold=0.05)
            wilcoxon_rank = (plot_cfg['wilcoxon'][key].sum(axis=1)).astype(int)
            plot_cfg['color'][key] = wilcoxon_rank
            text = np.array(wilcoxon_rank) / (len(wilcoxon_rank) - 1)
            # text = ['{:.2f}'.format(abs(t)) for t in text]
            plot_cfg['color_text'][key] = text

    template = 'plotly_white'
    for plot_cfg in plot_cfgs:
        if plot_cfg.get('type', 'box') == 'box':
            box_plot_colors(plot_cfg,
                            labels=lbls,
                            x_title=x_title,
                            color_label_pos=plot_cfg.get('color_position'),
                            y_title=plot_cfg.get('y_title', None),
                            secondary_y=plot_cfg.get('secondary_y', False),
                            quantile_thold=0.15,
                            show_footnote=False,
                            save=cfg['save_plots'],
                            label_scale=1.7,
                            size=(1000, 1000),
                            save_pdf=True,
                            file_path=os.path.join(img_folder, f'{file_cfg["fig"]}_{plot_cfg["name"]}')
                            )

    if file_cfg.get('color_per_subset'):
        try:
            exp_lbls = [d['type'] for d in cont_results['exp_lbl']]
            group_ix, j = 0, 0
            colors = [color_sequences[group_ix][j]]
            marker_centroids = [marker_names[0]]
            for i in range(1, len(exp_lbls)):
                if exp_lbls[i] != exp_lbls[i - 1]:
                    group_ix += 1
                    j = 0
                else:
                    j += 1
                colors.append(color_sequences[group_ix][j])
                marker_centroids.append(marker_names[group_ix])
        except Exception as e:
            colors = None
            marker_centroids = marker_names[:len(cont_results['exp_lbl'])]
    else:
        colors = None
        marker_centroids = marker_names[:len(cont_results['exp_lbl'])]

    # %%
    titles = ['Hypervolume', 'Distance', 'Predictor and corrector norm', 'Function and gradient evals',
              'Corrector and descent steps']
    dfs = [subset_metrics_dfs, subset_metrics_dfs, pred_corr_metrics_dfs, pred_corr_metrics_dfs, pred_corr_metrics_dfs]
    secondary_ys = [False, False, True, False, False]
    get_values = [{'train hv': {'mean': ['train', 'hv']},
                   'valid hv': {'mean': ['valid', 'hv']},
                   },
                  {'train dist': {'mean': ['train', 'mean norm'],
                                  'std': ['train', 'std norm'],
                                  'count': ['train', 'count']},
                   'valid dist': {'mean': ['valid', 'mean norm'],
                                  'std': ['valid', 'std norm'],
                                  'count': ['valid', 'count']},
                   },
                  {'predictor norm': {'mean': ['predictor', 'mean norm'],
                                      'std': ['predictor', 'std norm'],
                                      'count': ['predictor', 'count']},
                   'corrector norm': {'mean': ['corrector', 'mean norm'],
                                      'std': ['corrector', 'std norm'],
                                      'count': ['corrector', 'count']},
                   },
                  {'predictor f evals': {'mean': ['predictor', 'f_evals']},
                   'predictor g evals': {'mean': ['predictor', 'grad_evals']},
                   'corrector f evals': {'mean': ['corrector', 'f_evals']},
                   'corrector g evals': {'mean': ['corrector', 'grad_evals']},
                   'descent f evals': {'mean': ['descent', 'f_evals']},
                   'descent g evals': {'mean': ['descent', 'grad_evals']},
                   },
                  {'corrector steps': {'mean': ['corrector', 'mean per step'],
                                       'std': ['corrector', 'std per step'],
                                       'count': ['corrector', 'count']},
                   'descent steps': {'mean': ['descent', 'count']},
                   }
                  ]

    compiled_dfs = []
    for title, df, sy, value in zip(titles, dfs, secondary_ys, get_values):
        compiled_dict = get_compiled_dict(df, lbls, value)
        compiled_dict = adapt_runs(compiled_dict)
        compiled_df = df_from_dict(compiled_dict)
        pad_str_num_cols(compiled_df, round_dig=0)
        cols = np.unique([c.replace('_std', '') for c in compiled_df.columns])
        latex_df = concat_mean_std_df(compiled_df, cols)
        compiled_dfs.append(latex_df)
        print(tabulate(latex_df, headers='keys', tablefmt='psql'))

        if cfg['save_latex']:
            if title == 'Function and gradient evals':
                latex_df.drop(['predictor g evals', 'descent f evals', 'descent g evals'], axis=1, inplace=True)
            title = f'{title} varying {file_cfg["x_title"]}'
            write_text_file(os.path.join(results_folder, f"{title.replace(' ', '_')}"),
                            latex_table(title,
                                        latex_df.to_latex(escape=False, column_format='r' + 'r' * latex_df.shape[1])))

    all_dfs = [times_dfs] + dfs
    all_get_values = [{'f(x)': {'mean': ['f(x)', 'mean (s)']},
                       'J(x)': {'mean': ['J(x)', 'mean (s)']},
                       'execution time': {'mean': ['execution', 'mean (s)']}}]
    all_get_values += get_values
    compiled_dict, compiled_df = compile_metrics(lbls, all_dfs, all_get_values)

    tot_f = np.array(compiled_dict['predictor f evals']['mean']) + np.array(compiled_dict['corrector f evals']['mean'])
    tot_g = np.array(compiled_dict['predictor g evals']['mean']) + np.array(compiled_dict['corrector g evals']['mean'])
    factor = np.array(compiled_dict['J(x)']['mean']) / np.array(compiled_dict['f(x)']['mean'])

    # something is off if we get inf values
    factor[np.isinf(factor)] = -1.0

    weighted_f_evals = tot_f + tot_g * factor
    compiled_dict['weighted f evals'] = {'mean': weighted_f_evals}

    if isinstance(cont_results['exp_results'][0]['results'], list):
        mu = np.vstack([np.array(compiled_dict['train hv']['mean']), np.array(compiled_dict['valid hv']['mean'])]).T
        st = np.vstack([np.array(compiled_dict['train hv']['std']), np.array(compiled_dict['valid hv']['std'])]).T
        cnt = np.ones_like(mu) * len(subset_metrics_dfs[0])
        muc, stc = combine_means_stds(mu, st, cnt)
        compiled_dict['mean hv'] = {'mean': muc, 'std': stc}
    else:
        hvs = np.vstack([np.array(compiled_dict['train hv']['mean']), np.array(compiled_dict['valid hv']['mean'])])
        compiled_dict['mean hv'] = {'mean': np.mean(hvs, axis=0), 'std': np.std(hvs, axis=0)}

    compiled_df = df_from_dict(compiled_dict)
    compiled_df['factor'] = factor
    for key in ['f(x)', 'J(x)', 'f(x)_std', 'J(x)_std']:
        compiled_df[key] = compiled_df[key] * 1000

    # %%
    cols = ['descent steps', 'corrector steps']
    all_cols = cols + [c + '_std' for c in cols]
    steps_df = concat_mean_std_df(pad_str_num_cols(compiled_df.loc[:, all_cols], round_dig=1, inplace=False), cols)

    cols = ['predictor norm', 'corrector norm']
    all_cols = cols + [c + '_std' for c in cols]
    norms_df = concat_mean_std_df(pad_str_num_cols(compiled_df.loc[:, all_cols], round_dig=1, inplace=False), cols)

    cols = ['f(x)', 'J(x)', 'execution time', 'predictor f evals', 'corrector f evals', 'corrector g evals']
    all_cols = cols + [c + '_std' for c in cols]
    eval_df = concat_mean_std_df(pad_str_num_cols(compiled_df.loc[:, all_cols], round_dig=1, inplace=False), cols)

    cols = ['train hv', 'valid hv', 'mean hv']
    all_cols = cols + [c + '_std' for c in cols]
    hvs_df = concat_mean_std_df(pad_str_num_cols(compiled_df.loc[:, all_cols], round_dig=3, thold=1e-1, inplace=False),
                                cols)

    eval_df.rename(columns={'f(x)': 'time Fx (ms)', 'J(x)': 'time Jx (ms)'}, inplace=True)

    overview_df = pd.concat(
        [eval_df, hvs_df, steps_df, norms_df, compiled_df.loc[:, ['factor', 'weighted f evals']].round(2)], axis=1)

    print(overview_df.columns)
    f_evals_df = overview_df.loc[:,
                 ['execution time', 'time Fx (ms)', 'time Jx (ms)', 'factor', 'predictor f evals', 'corrector f evals',
                  'corrector g evals', 'weighted f evals']]

    f_evals2_df = overview_df.loc[:,
                  ['descent steps', 'corrector steps', 'predictor norm', 'corrector norm', 'train hv', 'valid hv']]
    time_evals_df = overview_df.loc[:,
                    ['execution time', 'weighted f evals', 'valid hv']]
    hvs_evals_df = overview_df.loc[:, ['train hv', 'valid hv']]

    save_df_to_latex(f_evals_df, 'Time execution and evaluations count',
                     os.path.join(results_folder, "f_evals_overview"))
    save_df_to_latex(time_evals_df, 'General overview', os.path.join(results_folder, "time_evals_hvs_overview"))
    save_df_to_latex(hvs_evals_df, 'Hypervolume results', os.path.join(results_folder, "hvs_overview"))
    save_df_to_latex(f_evals2_df, 'Steps count, norm and hypervolume results',
                     os.path.join(results_folder, "f_evals2_overview"))

    print_df = compiled_df.loc[:, ['execution time', 'weighted f evals', 'mean hv']]
    print(tabulate(print_df, headers='keys', tablefmt='psql'))

    plot_dict = {}
    for key in ['lbls', 'execution time', 'weighted f evals', 'mean hv']:
        plot_dict[key] = compiled_dict[key]

    bar_plot_3axes_with_errors(plot_dict, title='Cpu time, weighted f evals, and hv')

    # %%
    data = np.array([compiled_dict['weighted f evals']['mean'] / 1000, compiled_dict['valid hv']['mean'],
                     compiled_dict['execution time']['mean']]).T
    performance = pd.DataFrame(data, columns=['weighted f evals', 'valid hv', 'execution time'], index=lbls)
    evals_max, et_max = 3.9, 1200

    performance_norm = pd.DataFrame()
    performance_norm['norm weighted f evals'] = (performance['weighted f evals'] / evals_max) * 100
    performance_norm['norm exec time'] = (performance['execution time'] / et_max) * 100
    performance_norm['norm HV'] = ((performance['valid hv'] - hv_min) / (hv_max - hv_min)) * 100
    print(tabulate(performance_norm, headers='keys', tablefmt='psql'))

    # %%
    keys = ['corrector g evals', 'predictor g evals', 'corrector f evals', 'predictor f evals']

    fg_evals = {key: np.array(results_items[key]) for key in keys}
    tot_fg_evals = {'tot_g': fg_evals['predictor g evals'] + fg_evals['corrector g evals'],
                    'tot_f': fg_evals['predictor f evals'] + fg_evals['corrector f evals']}

    weighted_f = [f + g * c for f, g, c in zip(tot_fg_evals['tot_f'], tot_fg_evals['tot_g'], factor)]

    keys = ['train hv', 'valid hv']

    hvs = {key: np.array(results_items[key]) for key in keys}
    mean_hv = [(t + v) / 2 for t, v in zip(hvs['valid hv'], hvs['valid hv'])]

    norm_f = np.array([(f * 100) / (evals_max * 1000) for f in weighted_f])
    norm_hv = np.array([(h - hv_min) / (hv_max - hv_min) * 100 for h in mean_hv])
    norm_et = np.array([(np.array(r) / et_max) * 100 for r in results_items['exec time']])

    performance_norm = pd.DataFrame()

    keys = ['norm weighted f evals', 'norm exec time', 'norm mean HVs']
    for key, arr in zip(keys, [norm_f, norm_et, norm_hv]):
        performance_norm[key] = np.mean(arr, axis=-1)
        performance_norm[key + '_std'] = np.std(arr, axis=-1)

    performance_norm.index = lbls

    # %%
    pad_str_num_cols(performance_norm, round_dig=2)
    print(performance_norm)
    cols = np.unique([c.replace('_std', '') for c in performance_norm.columns])
    latex_df = concat_mean_std_df(performance_norm, cols)
    print(tabulate(latex_df, headers='keys', tablefmt='psql'))

    if cfg['save_latex']:
        write_text_file(os.path.join(results_folder, 'main_table'),
                        latex_table('Results comparison', latex_df.to_latex(escape=False)))

    plot_2d_grouped_traces([norm_f, norm_hv],
                           names=lbls,
                           markersizes=12,
                           colors=colors,
                           markersymbols=None,
                           centroidsymbols=marker_centroids,
                           axes_labels=['Normalized weighted function evaluations', 'Normalized valid HV'],
                           file_path=os.path.join(img_folder, f'eval_{file_cfg["fig"]}'),
                           save=cfg['save_plots'],
                           save_pdf=True,
                           size=(1000, 900),
                           # title=performance_title,
                           legend_title=performance_title,
                           )

    if isinstance(cont_results['exp_results'][0]['results'], list):
        Fs = [sort_arr_1st_col(res['results'][0]['population']['F']) for res in cont_results['exp_results']]
        fx_inis = [res['results'][0]['independent'][0]['descent']['ini_fx'].reshape(1, -1) for res in
                   cont_results['exp_results']]
    else:
        Fs = [sort_arr_1st_col(res['results']['population']['F']) for res in cont_results['exp_results']]
        fx_inis = [res['results']['independent'][0]['descent']['ini_fx'].reshape(1, -1) for res in
                   cont_results['exp_results']]
    names = lbls + [l + '_ini' for l in lbls]

    if file_cfg.get('color_per_subset'):
        ix_mask = list(range(0, 5))
        Fs = [Fs[i] for i in ix_mask]
        fx_inis = [fx_inis[i] for i in ix_mask]
        colors = [colors[i] for i in ix_mask]
        names = [lbls[i] for i in ix_mask] + [l + '_ini' for l in [lbls[i] for i in ix_mask]]

    plot_2D_pf(Fs, fx_inis, names, cfg['save_plots'], os.path.join(img_folder, 'pfs'),
               f_markersize=5,
               f_mode='markers+lines',
               colors_ixs=colors * 2 if colors is not None else None,
               label_scale=1.8,
               legend_title=file_cfg['x_title'])


if __name__ == '__main__':
    project = 'snp'

    cfg = {'save_plots': False,
           'save_results': False,
           'save_latex': False,
           'plot_title': False}

    ## Run single optimization with SDS and compare it with MOEAS
    run_sds_vs_moea_optimization(project, cfg, run_moeas=False)

    ## Code to run the experiments
    for experiment in ['condition', 'model_size', 'step_eps', 't_eps', 'batch_size']:
        run_sds_experiment(project, cfg, experiment, n_runs=2)

    # Explore experiment results
    experiment_results_cfg = [
        {'key': 'step_eps', 'folder': 'step_eps', 'experiment': 'standalone_ix_4_it',
         'prefix': 'step_eps:', 'x_title': 'step size', 'fig': 'steps'},
        {'key': 'condition', 'folder': 'type_in_pf_eps_type_in_pf_eps_type_in_pf_eps',
         'experiment': 'standalone_ix_15_it',
         'prefix': ['type:', ('in_pf_eps:', ': '), ', ', ('delta', 'd'), ('projection', 'p'), ('rank', 'r')],
         'x_title': 'stop criteria', 'color_per_subset': True, 'fig': 'criteria'},
        {'key': 't_eps', 'folder': 'eps', 'experiment': 'standalone_ix_9_it', 'prefix': 'eps:',
         'x_title': 'beta', 'fig': 'beta'},
        {'key': 'batch_size', 'folder': 'moo_batch_size', 'experiment': 'standalone_ix_8_it',
         'prefix': 'moo_batch_size:',
         'x_title': 'batch size', 'fig': 'batch'},
        {'key': 'model_size', 'folder': 'split_model', 'experiment': 'standalone_ix_2_it', 'prefix': 'split_model:',
         'x_title': 'MOP problem', 'fig': 'problem'},
    ]

    # get hv min and max for normalized plots
    hv_valid = []
    for experiment in ['condition', 'model_size', 'step_eps', 't_eps', 'batch_size']:
        file_cfg = [c for c in experiment_results_cfg if c['key'] == experiment][0]
        cont_results = joblib.load(
            os.path.join(get_result_folder({}, project), 'experiments',
                         file_cfg['folder'], file_cfg['experiment']) + '.z')

        for exp in cont_results['exp_results']:
            for run in exp['metrics']:
                hv_valid.append(run['subset_metrics'].loc['valid', 'hv'])

    # get tables and plots from results
    hv_max, hv_min = math.ceil(np.max(hv_valid) * 10) / 10, math.floor(np.min(hv_valid) * 10) / 10
    for experiment in ['condition', 'model_size', 'step_eps', 't_eps', 'batch_size']:
        explore_experiment_results(project, cfg, experiment, experiment_results_cfg, hv_min, hv_max)


