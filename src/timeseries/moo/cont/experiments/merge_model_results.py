import math
import os
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
from plotly.graph_objs import Figure
from tabulate import tabulate

from src.timeseries.moo.cont.core.config import params_cfg, experiments_cfg, cont_cfg
from src.timeseries.moo.cont.core.harness import plot_2D_pf, get_cfg_from_loop_cfg, get_sublevel_keys
from src.timeseries.moo.cont.utils.results import get_compiled_dict, df_from_dict, compile_metrics, combine_means_stds, \
    adapt_runs
from src.timeseries.moo.cont.utils.util import get_from_dict
from src.timeseries.utils.dataset import load_file
from src.timeseries.utils.filename import get_result_folder
from src.timeseries.utils.files import save_vars
from src.timeseries.utils.moo import sort_arr_1st_col
from src.utils.plot import plot_2D_points_traces_total, plot_2D_predictor_corrector, plot_bidir_2D_points_vectors, \
    plotly_colors, plotly_save, plot_pfs, bar_plots_with_errors, bar_plot_3axes_with_errors, plot_2D_points_traces, \
    plot_boxes
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import itertools

pio.renderers.default = "browser"

if __name__ == '__main__':
    # %%
    # general_cfg = {'save_plot': False,
    #                'save_results': False,
    #                'show_title': False,
    #                'plot_individual_pf': False,
    #                'color_per_subset': True,
    #                }

    project = 'snp'

    folder = 'results'
    file = '{}_ix_1_it_2'
    exp_cfg_name = 'model_ix'
    output_filename = '10_it_5'

    base_path = os.path.join(get_result_folder({}, project), 'experiments')
    cont_results = [
        joblib.load(os.path.join(get_result_folder({}, project), 'experiments', folder, file.format(i)) + '.z')
        for i in range(1, 12)]

    # folder = 'split_model'
    # files = ['2_ix_1_it_2', '2_ix_1_it']
    #
    # base_path = os.path.join(get_result_folder({}, project), 'experiments')
    # cont_results = [
    #     joblib.load(os.path.join(get_result_folder({}, project), 'experiments', folder, file + '.z'))
    #     for file in files]

    # %%
    results = [res['exp_results'][0] for res in cont_results]
    exp_cfg = experiments_cfg[exp_cfg_name]
    relevant_cfg = [params_cfg[k] for k in get_sublevel_keys(exp_cfg['loop_cfg'], [])]
    cfgs = get_cfg_from_loop_cfg(exp_cfg['loop_cfg'], params_cfg, cont_cfg, [])
    experiment_labels = [dict([(c['keys'][-1], get_from_dict(cfg, c['keys'])) for c in relevant_cfg]) for cfg in cfgs]

    # %%

    save_vars({'params_cfg': params_cfg,
               'exp_cfg': exp_cfg,
               'exp_lbl': experiment_labels,
               'exp_results': results},
              os.path.join(base_path,
                           folder,
                           output_filename),
              )
