import os

import joblib
import numpy as np
import seaborn as sns

from src.models.compare.winners import Winners
from src.timeseries.plot.moo import plot_runs, plot_boxplot, plot_2D_moo_dual_history
from src.timeseries.utils.filename import get_result_folder
from src.timeseries.utils.files import save_vars
from src.timeseries.utils.results import compile_multiple_results, get_hv_results_from_runs
from src.timeseries.utils.util import write_latex_from_scores, write_text_file, latex_table

sns.set_theme('poster')
sns.set_style("dark")

if __name__ == "__main__":
    # %%
    general_cfg = {'save_plot': True,
                   'save_results': False,
                   'show_title': True,
                   'comparison_name': 'seeding_ES_ema_r_q258_g100_p100_c1_eq0',
                   }

    project = 'snp'

    weights_files = [
        ('60t_ema_q159', 'TFTModel_ES_ema_r_q159_NSGA2_g100_p100_s0_c1_eq0_dual_wmoo_repeat5'),
        ('60t_ema_q159', 'TFTModel_ES_ema_r_q159_NSGA2_g100_p100_s1_c1_eq0_dual_wmoo_repeat5'),
    ]

    results_folder = os.path.join(get_result_folder({}, project), 'compare', general_cfg['comparison_name'])
    moo_results = [joblib.load(os.path.join(get_result_folder({}, project), file[0], 'moo', file[1]) + '.z')
                   for file in weights_files]

    for i in range(len(moo_results)):
        if not isinstance(moo_results[i], list):
            moo_results[i] = [moo_results[i]]

    experiment_labels = ['sampling: {}'.format(experiment[0]['lq']['algo_cfg']['use_sampling']) for experiment in
                         moo_results]

    results = compile_multiple_results(moo_results, experiment_labels, hv_ref=[10] * 2)

    # %% HV results
    q_exp_hvs, hvs_df = get_hv_results_from_runs(results, experiment_labels)

    # %% Winners
    metric = np.negative(np.mean(q_exp_hvs, axis=2))
    winners = Winners(metric, experiment_labels)
    scores = winners.score(q_exp_hvs, alternative='greater')

    # %% Grouped runs per quantile
    q_exp_hist, q_exp_mean, all_plot_labels, q_exp_last = [], [], [], []
    for q_lbl, q_res in results.items():
        y_runs = []
        for exp_lbl, exp_res in q_res['hv_hist'].items():
            exp_res_mod = np.array(exp_res)[:, 1:]
            q_exp_last.append(exp_res_mod[:, -1])
            q_exp_mean.append(np.mean(exp_res_mod, axis=0))
            y_runs.append(np.mean(exp_res_mod, axis=0))
            all_plot_labels.append('{} {}'.format(exp_lbl, 'lq' if q_lbl == 'lower quantile' else 'uq'))

        q_exp_hist.append(np.array(y_runs))

    q_exp_mean = np.array(q_exp_mean)
    q_exp_last = np.array(q_exp_last)

    # %% plot all

    plot_runs(q_exp_hist,
              mean_run=None,
              x_label='Generation',
              y_label='Hypervolume',
              title='HV history',
              size=(15, 9),
              file_path=os.path.join(results_folder, 'img', 'hv_history'),
              save=general_cfg['save_plot'],
              legend_labels=all_plot_labels,
              show_grid=True,
              show_title=general_cfg['show_title'])
    #
    plot_boxplot(q_exp_last,
                 all_plot_labels,
                 x_label='Algorithm',
                 y_label='Hypervolume',
                 title='Hypervolume for quantiles',
                 size=(15, 9),
                 # ylim=(96, 99),
                 file_path=os.path.join(results_folder, 'img', 'hv_box_plot'),
                 save=general_cfg['save_plot'],
                 show_grid=True,
                 show_title=general_cfg['show_title'])

    # %% plot history of only one run
    run_selected = 0
    plot_gens = list(range(0, 25, 5))

    original_losses, history = {}, {}
    for q_lbl, res in results.items():
        original_losses[q_lbl] = {}
        history[q_lbl] = {}
        for exp_lbl, sub_res in res['original_losses'].items():
            original_losses[q_lbl][exp_lbl] = sub_res[run_selected]
        for exp_lbl, sub_res in res['history'].items():
            # remove first generation because constraints
            # are applied after the first random generation
            history[q_lbl][exp_lbl] = sub_res[run_selected]

    for q_lbl in results:
        for exp_lbl in experiment_labels:
            filename = '{}_{}_history'.format(exp_lbl.replace(" ", "_").replace(":",""), q_lbl.replace(" ", "_"))
            Fs = history[q_lbl][exp_lbl]
            o_l = original_losses[q_lbl][exp_lbl]
            plot_2D_moo_dual_history(Fs,
                                     remove_1st_gen=True,
                                     save=general_cfg['save_plot'],
                                     plot_gens=plot_gens,
                                     file_path=os.path.join(results_folder, 'img', filename),
                                     original_loss=o_l,
                                     figsize=(50, 20),
                                     n_snapshots=5,
                                     title='History',
                                     markersize=5,
                                     plot_title=True)

    # %%
    if general_cfg['save_results']:
        results['weights_files'] = weights_files
        save_vars(results, os.path.join(results_folder,
                                        '{}'.format(general_cfg['comparison_name'])))

        write_latex_from_scores(scores,
                                os.path.join(results_folder,
                                             'txt',
                                             '{}_scores'.format(general_cfg['comparison_name'])))

        write_text_file(os.path.join(results_folder,
                                     'txt',
                                     '{}'.format(general_cfg['comparison_name'])),
                        latex_table('Hypervolume for quantiles', hvs_df.to_latex()))
