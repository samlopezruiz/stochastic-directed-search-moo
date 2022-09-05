import gc
import os

import numpy as np
from numba import cuda
from tabulate import tabulate

from src.timeseries.moo.cont.core.config import cont_cfg
from src.timeseries.moo.cont.core.harness import get_model_and_params, get_ts_problem, get_continuation_method, \
    run_cont_problem, save_cont_resuls, get_cfg_from_loop_cfg, \
    get_sublevel_keys, plot_pf_and_total
from src.timeseries.moo.cont.utils.util import set_in_dict, get_from_dict
from src.timeseries.utils.files import save_vars
import tensorflow as tf

from keras import backend as K


def run_experiments(cfgs, get_model, get_problem, get_cont, use_gpu=True):
    exp_results = []
    for i, cfg in enumerate(cfgs):

        print('{}/{} Experiment: {}'.format(i + 1,
                                            len(cfgs),
                                            dict([(c['keys'][-1], get_from_dict(cfg, c['keys'])) for c in
                                                  relevant_cfg])))

        if i == 0:
            model_params, results_folder = get_model_and_params(cfg, project, use_gpu=use_gpu)
            problem = get_ts_problem(cfg, model_params, test_ss=False, use_gpu=use_gpu)
            ds_cont = get_continuation_method(cfg, problem)

        if get_model and i > 0:
            print('resetting gpu memory...')
            model_params['opt_manager'].hyperparam_folder = model_params['opt_manager'].hyperparam_folder[
                                                            :-1] + str(
                get_from_dict(cfg, ['model', 'ix']))
            model_params['model'].load(model_params['opt_manager'].hyperparam_folder, use_keras_loadings=True)

        if (get_model or get_problem) and i > 0:
            problem = get_ts_problem(cfg, model_params, test_ss=False, use_gpu=use_gpu)

        if (get_model or get_cont) and i > 0:
            ds_cont = get_continuation_method(cfg, problem)

        # reset batch ix
        problem.train_batch_ix = 0
        results, metrics = run_cont_problem(ds_cont, problem)

        exp_results.append({'results': results, 'metrics': metrics})

    # Save results
    model_ix = get_from_dict(cont_cfg, ['model', 'ix'])
    filename = '{}_ix_{}_it'.format(model_ix, len(cfgs))
    save_vars({'params_cfg': params_cfg,
               'loop_cfg': loop_cfg,
               'exp_lbl': experiment_labels,
               'exp_results': exp_results},
              os.path.join(os.path.dirname(results_folder),
                           'experiments',
                           general_cfg['experiment_name'],
                           filename),
              )

    return exp_results


# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

if __name__ == '__main__':
    # %%
    general_cfg = {'save_plots': False,
                   'save_results': True,
                   'plot_title': False,
                   'experiment_name': 'model_ix',
                   }

    project = 'snp'

    set_in_dict(cont_cfg, ['model', 'ix'], 2)
    set_in_dict(cont_cfg, ['corrector', 'batch_gradient'], True)
    set_in_dict(cont_cfg, ['cont', 'verbose'], False)
    # set_in_dict(cont_cfg, ['cont', 'step_eps'], 0.02)

    params_cfg = {'steps': {'keys': ['cont', 'step_eps'],
                            'values': np.round(np.arange(0.005, 0.04, 0.01), 4)},
                  'batch_size': {'keys': ['problem', 'moo_batch_size'],
                                 'values': [2 ** i for i in range(1, 6)]},
                  'model': {'keys': ['model', 'ix'],
                            'values': list(range(5, 7))},
                  'batch_gradient': {'keys': ['corrector', 'batch_gradient'],
                                     'values': [True, False]},
                  }

    # loop_cfg = {'steps':
    #                 {'batch_size':
    #                      {}
    #                  }
    #             }

    loop_cfg = {'model': {}}

    relevant_cfg = [params_cfg[k] for k in get_sublevel_keys(loop_cfg, [])]
    cfgs = get_cfg_from_loop_cfg(loop_cfg, params_cfg, cont_cfg, [])
    experiment_labels = [dict([(c['keys'][-1], get_from_dict(cfg, c['keys'])) for c in relevant_cfg]) for cfg in
                         cfgs]

    print('-----EXPERIMENTS-----')
    header = experiment_labels[0].keys()
    rows = [x.values() for x in experiment_labels]
    print(tabulate(rows, header, tablefmt='psql'))

    # %%#%%
    exp_results = run_experiments(cfgs,
                                  get_model=False,
                                  get_problem=False,
                                  get_cont=True,
                                  use_gpu=True)


