from tabulate import tabulate

from src.timeseries.moo.cont.core.config import cont_cfg
from src.timeseries.moo.cont.core.harness import get_model_and_params, get_ts_problem, get_continuation_method, \
    run_cont_problem, plot_pf_and_total, save_cont_resuls, save_latex_table
from src.timeseries.moo.cont.utils.bash import get_input_args
from src.timeseries.moo.cont.utils.util import set_in_dict, get_from_dict
from src.timeseries.utils.moo import sort_1st_col
from src.utils.plot import plot_2D_points_traces_total

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
    print('Model ix: {}'.format(get_from_dict(cont_cfg, ['model', 'ix'])))
    set_in_dict(cont_cfg, ['cont', 'step_eps'], 1e-2)
    set_in_dict(cont_cfg, ['problem', 'split_model'], 'medium')

    model_params, results_folder = get_model_and_params(cont_cfg, project)
    problem = get_ts_problem(cont_cfg, model_params, test_ss=False)
    ds_cont = get_continuation_method(cont_cfg, problem)

    results, metrics = run_cont_problem(ds_cont, problem)

    # Save results
    save_cont_resuls({'results': results, 'metrics': metrics, 'cont_cfg': cont_cfg}, results_folder, cfg, cont_cfg)

    # Save latex tables
    save_latex_table(metrics, results_folder, cfg, cont_cfg)

    # Plot results
    plot_pf_and_total(results, results_folder, cfg, cont_cfg)

    # %%
    X_sorted, F_sorted = sort_1st_col(results['population']['X'], results['population']['F'])
    fx_ini = results['independent'][0]['descent']['ini_fx'].reshape(1, 2)

    plot_2D_points_traces_total([F_sorted, fx_ini],
                                names=['cont', 'ini'],
                                color_ixs=[0, 10],
                                markersizes=[10, 20],
                                marker_symbols=['circle', 'star'],
                                modes=['markers+lines', 'markers'],
                                outlines=[False, True],
                                axes_labels=('Objective 1', 'Objective 2'),
                                label_scale=1.8)

    # %% Print metrics
    print(tabulate(metrics['pred_corr_metrics'], headers='keys', tablefmt='psql'))
    print(tabulate(metrics['subset_metrics'], headers='keys', tablefmt='psql'))
    print(tabulate(metrics['times'], headers='keys', floatfmt=(None, ",.4f"), tablefmt='psql'))

    # #%%
    # self = problem
    # import tensorflow as tf
    # import numpy as np
    #
    # x = tf.convert_to_tensor(np.arange(5))
    # y = np.array(['a', 'b', 'c', 'd', 'e'])
    #
    # indices = tf.range(start=0, limit=tf.shape(self.moo_model_input_train_unbatched)[0], dtype=tf.int32)
    # shuffled_indices = tf.random.shuffle(indices)
    #
    # shuffled_x = tf.gather(self.moo_model_input_train_unbatched, shuffled_indices)
    # shuffled_y = self.y_train[shuffled_indices, Ellipsis]
