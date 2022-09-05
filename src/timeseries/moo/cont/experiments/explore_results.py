import math
import os
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
from plotly.graph_objs import Figure
from tabulate import tabulate

from src.timeseries.moo.cont.core.harness import plot_2D_pf
from src.timeseries.moo.cont.utils.results import get_compiled_dict, df_from_dict, compile_metrics
from src.timeseries.moo.cont.utils.util import get_from_dict
from src.timeseries.utils.dataset import load_file
from src.timeseries.utils.filename import get_result_folder
from src.timeseries.utils.moo import sort_arr_1st_col
from src.utils.plot import plot_2D_points_traces_total, plot_2D_predictor_corrector, plot_bidir_2D_points_vectors, \
    plotly_colors, plotly_save, plot_pfs, bar_plots_with_errors, bar_plot_3axes_with_errors, plot_2D_points_traces
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

pio.renderers.default = "browser"

if __name__ == '__main__':
    general_cfg = {'save_plot': False,
                   'save_results': False,
                   'show_title': False,
                   'experiment_name': 'model_ix',
                   'plot_individual_pf': False,
                   'color_per_subset': True,
                   }

    project = 'snp'

    files = [
        # ('moo_batch_size', '2_ix_8_it_1'),
        # ('moo_batch_size', '2_ix_8_it_1'),
        ('type', '2_ix_1_it'),
        # ('moo_batch_size', '2_ix_8_it_1'),
        # ('type_in_pf_eps_type_in_pf_eps_type_in_pf_eps', '2_ix_18_it'),
    ]

    base_path = os.path.join(get_result_folder({}, project), 'experiments', general_cfg['experiment_name'])
    # results_folder = os.path.join(project, 'compare', general_cfg['experiment_name'])
    cont_results = [joblib.load(os.path.join(get_result_folder({}, project), 'experiments', file[0], file[1]) + '.z')
                    for file in files]
    cont_results = cont_results[0]
    lbls = ['_'.join(['{}_{}'.format(k, v) for k, v in d.items()]) for d in cont_results['exp_lbl']]

    # %% Use only one experiment file
    subset_metrics_dfs = [get_from_dict(res, ['metrics', 'subset_metrics']) for res in cont_results['exp_results']]
    pred_corr_metrics_dfs = [get_from_dict(res, ['metrics', 'pred_corr_metrics']) for res in
                             cont_results['exp_results']]
    times_dfs = [get_from_dict(res, ['metrics', 'times']) for res in cont_results['exp_results']]

    # %%
    if general_cfg['color_per_subset']:
        exp_lbls = [d['type'] for d in cont_results['exp_lbl']]
        counter = 0
        colors_ixs = [0]
        for i in range(1, len(exp_lbls)):
            if exp_lbls[i] != exp_lbls[i - 1]:
                counter += 1
            colors_ixs.append(counter)
    else:
        colors_ixs = None

    # %%
    # Hv
    titles = ['Hypervolume', 'Distance', 'Predictor and corrector norm', 'Function and gradient evals',
              'Corrector steps']
    dfs = [subset_metrics_dfs, subset_metrics_dfs, pred_corr_metrics_dfs, pred_corr_metrics_dfs, pred_corr_metrics_dfs]
    secondary_ys = [False, False, True, False, False]
    get_values = [{'train hv': {'mean': ['train', 'hv']},
                   'valid hv': {'mean': ['valid', 'hv']},
                   },
                  {'train dist': {'mean': ['train', 'mean norm'],
                                  'std': ['train', 'std norm']},
                   'valid dist': {'mean': ['valid', 'mean norm'],
                                  'std': ['valid', 'std norm']},
                   },
                  {'predictor norm': {'mean': ['predictor', 'mean norm'],
                                      'std': ['predictor', 'std norm']},
                   'corrector norm': {'mean': ['corrector', 'mean norm'],
                                      'std': ['corrector', 'std norm']},
                   },
                  {'predictor f evals': {'mean': ['predictor', 'f_evals']},
                   'predictor g evals': {'mean': ['predictor', 'grad_evals']},
                   'corrector f evals': {'mean': ['corrector', 'f_evals']},
                   'corrector g evals': {'mean': ['corrector', 'grad_evals']},
                   },
                  {'corrector steps': {'mean': ['corrector', 'mean per step'],
                                       'std': ['corrector', 'std per step']},
                   }
                  ]

    for title, df, sy, value in zip(titles, dfs, secondary_ys, get_values):
        compiled_dict = get_compiled_dict(df, lbls, value)
        compiled_df = df_from_dict(compiled_dict)
        print(tabulate(compiled_df, headers='keys', tablefmt='psql'))
        bar_plots_with_errors(compiled_dict, title=title, secondary_y=sy)

    # %%
    # Exec Time
    get_values = [{'f(x)': {'mean': ['f(x)', 'mean (s)']},
                   'J(x)': {'mean': ['J(x)', 'mean (s)']},
                   'execution time': {'mean': ['execution', 'mean (s)']}},
                  {'predictor f evals': {'mean': ['predictor', 'f_evals']},
                   'predictor g evals': {'mean': ['predictor', 'grad_evals']},
                   'corrector f evals': {'mean': ['corrector', 'f_evals']},
                   'corrector g evals': {'mean': ['corrector', 'grad_evals']},
                   },
                  {'train hv': {'mean': ['train', 'hv']},
                   'valid hv': {'mean': ['valid', 'hv']},
                   },
                  ]

    compiled_dict, compiled_df = compile_metrics(lbls, [times_dfs, pred_corr_metrics_dfs, subset_metrics_dfs],
                                                 get_values)

    tot_f = np.array(compiled_dict['predictor f evals']['mean']) + np.array(compiled_dict['corrector f evals']['mean'])
    tot_g = np.array(compiled_dict['predictor g evals']['mean']) + np.array(compiled_dict['corrector g evals']['mean'])
    factor = np.array(compiled_dict['J(x)']['mean']) / np.array(compiled_dict['f(x)']['mean'])
    weighted_f_evals = tot_f + tot_g * factor
    mean_hv = np.mean(
        np.vstack([np.array(compiled_dict['train hv']['mean']), np.array(compiled_dict['valid hv']['mean'])]), axis=0)
    compiled_dict['weighted f evals'] = {}
    compiled_dict['weighted f evals']['mean'] = np.round(weighted_f_evals, 0).astype(int)
    compiled_dict['mean hv'] = {}
    compiled_dict['mean hv']['mean'] = mean_hv

    compiled_df = df_from_dict(compiled_dict)

    print_df = compiled_df.loc[:, ['execution time', 'weighted f evals', 'mean hv']]
    print(tabulate(print_df, headers='keys', tablefmt='psql'))

    plot_dict = {'lbls': lbls,
                 'execution time': {'mean': compiled_dict['execution time']['mean']},
                 'weighted f evals': {'mean': compiled_dict['weighted f evals']['mean']},
                 'hv': {'mean': compiled_dict['mean hv']['mean']},
                 }
    bar_plot_3axes_with_errors(plot_dict, title='Cpu time, weighted f evals, and hv')

    # %% Performance
    performance = (compiled_dict['mean hv']['mean'] / max(compiled_dict['mean hv']['mean'])) * 100 - (
            compiled_dict['weighted f evals']['mean'] / max(compiled_dict['weighted f evals']['mean']))

    performance = performance / max(performance)

    plot_dict = {'lbls': lbls,
                 'performance': {
                     'mean': performance},
                 }
    bar_plots_with_errors(plot_dict, title='Performance', secondary_y=False)

    data = np.array([compiled_dict['weighted f evals']['mean'] / 1000, compiled_dict['mean hv']['mean']]).T
    data = [d.reshape(1, -1) for d in data]
    plot_2D_points_traces(data, names=lbls, color_ixs=colors_ixs)

    # %%
    img_path = os.path.join(base_path, 'img')
    Fs = [sort_arr_1st_col(res['results']['population']['F']) for res in cont_results['exp_results']]
    fx_inis = [res['results']['independent'][0]['descent']['ini_fx'].reshape(1, -1) for res in
               cont_results['exp_results']]
    names = lbls + [l + '_ini' for l in lbls]

    plot_2D_pf(Fs, fx_inis, names, general_cfg['save_plot'], os.path.join(img_path, 'pfs'),
               f_markersize=5,
               f_mode='markers+lines',
               colors_ixs=colors_ixs * 2 if colors_ixs is not None else None)

    # plot_pfs(Fs, fx_inis, lbls)

    # %% Plot individual
    if general_cfg['plot_individual_pf']:
        plot_results = [[p['population'] for p in res['results']['independent']] for res in cont_results['exp_results']]
        plot_descent = [[p['descent'] for p in res['results']['independent']] for res in cont_results['exp_results']]

        for res, descent in zip(plot_results, plot_descent):
            plot_bidir_2D_points_vectors(res,
                                         descent=descent,
                                         markersize=6,
                                         plot_ps=False,
                                         )
