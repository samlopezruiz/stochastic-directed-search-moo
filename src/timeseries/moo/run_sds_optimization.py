import os

import numpy as np
import pandas as pd
from tabulate import tabulate

from src.timeseries.moo.sds.config import sds_cfg
from src.timeseries.moo.core.harness import get_model_and_params, get_ts_problem, get_continuation_method, \
    run_cont_problem, save_cont_resuls, save_latex_table, plot_pf_and_total, filename_from_cfg
from src.timeseries.moo.sds.utils.bash import get_input_args
from src.timeseries.moo.sds.utils.util import get_from_dict, set_in_dict
from src.timeseries.utils.moo import sort_1st_col
from src.utils.plot import plot_2D_points_traces_total, plotly_colors, plotly_save

if __name__ == '__main__':
    # %%
    input_args = get_input_args()

    cfg = {'save_plots': False,
           'save_results': True,
           'save_latex': True,
           'plot_title': False,
           }

    project = 'snp'
    model = 'standalone'

    set_in_dict(sds_cfg, ['model', 'ix'], input_args['model_ix'])
    set_in_dict(sds_cfg, ['model', 'ix'], model)
    # set_in_dict(sds_cfg, ['sds', 'step_eps'], 5e-3)
    # set_in_dict(sds_cfg, ['sds', 'max_increment'], 0.05)
    print('Model ix: {}'.format(get_from_dict(sds_cfg, ['model', 'ix'])))

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

    print(tabulate(metrics['subset_metrics'], headers='keys', tablefmt='psql'))

