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

    # %%
    ## FOR DEMO PURPOSES
    def create_square(v1, v3):
        # Extract coordinates from the vertices
        x1, y1 = v1
        x3, y3 = v3

        # Return the vertices in a sequence for plotting
        return [x1, x1, x3, x3, x1], [y1, y3, y3, y1, y1]

    filename = filename_from_cfg(sds_cfg)
    img_path = os.path.join(results_folder, 'sds', 'img', filename)
    X_sorted, F_sorted = sort_1st_col(results['population']['X'], results['population']['F'])
    fx_ini = results['independent'][0]['descent']['ini_fx'].reshape(1, 2)
    fig = plot_2D_points_traces_total([F_sorted, fx_ini],
                                      names=['sds', 'ini'],
                                      color_ixs=[0, 10],
                                      show_legends=[False, True],
                                      markersizes=[8, 20],
                                      marker_symbols=['circle', 'star'],
                                      modes=['markers+lines', 'markers'],
                                      save=cfg['save_plots'],
                                      save_pdf=True,
                                      size=(1000, 900),
                                      outlines=[False, True],
                                      label_scale=1.7,
                                      file_path=img_path + '_total',
                                      axes_labels=('Objective 1', 'Objective 2'),
                                      return_fig=True)

    tot_max = np.sum(fx_ini) * 1.05

    square_vertexes = (pd.DataFrame(F_sorted)
                       .assign(tot=lambda df: df.sum(axis=1))
                       .assign(change=lambda df: np.sign(df.tot - tot_max).diff().fillna(0) != 0)
                       .pipe(lambda df: df[df.change].drop(['tot', 'change'], axis=1)))
    x, y = create_square(square_vertexes.iloc[0], square_vertexes.iloc[1])
    fig.add_scatter(x=x, y=y, mode='lines', marker=dict(color=plotly_colors[1]), showlegend=False, row=1, col=1)
    x, y = create_square([square_vertexes.iloc[0, 0], np.sum(F_sorted, axis=1).min()], [square_vertexes.iloc[1, 0], tot_max])
    fig.add_scatter(x=x, y=y, mode='lines', marker=dict(color=plotly_colors[1]), showlegend=False, row=1, col=2)

    fig.show()

    plotly_save(fig, os.path.join(img_path, 'example'), size=(1000, 750), save_pdf=True)

