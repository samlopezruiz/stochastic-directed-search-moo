import os

import numpy as np
from tabulate import tabulate

from src.timeseries.moo.sds.config import sds_cfg, params_cfg, experiments_cfg
from src.timeseries.moo.core.harness import get_cfg_from_loop_cfg, \
    get_sublevel_keys, run_experiments
from src.timeseries.moo.sds.utils.bash import get_input_args
from src.timeseries.moo.sds.utils.util import set_in_dict, get_from_dict
from src.timeseries.utils.files import save_vars

if __name__ == '__main__':
    # %%
    input_args = get_input_args()
    general_cfg = {'save_plots': True,
                   'save_results': True,
                   }

    project = 'snp'
    experiment = 'batch_size'
    model = 2

    exp_cfg = experiments_cfg[experiment]

    ##### (for demo purposes) uncomment this line #####
    # exp_cfg['ini_cfg']['seeds'] = np.arange(0, 3)
    ###############################################

    set_in_dict(sds_cfg, ['model', 'ix'], input_args['model_ix'])
    set_in_dict(sds_cfg, ['model', 'ix'], model)
    set_in_dict(sds_cfg, ['sds', 'verbose'], False)

    print('Model ix: {}'.format(sds_cfg['model']['ix']))

    relevant_cfg = [params_cfg[k] for k in get_sublevel_keys(exp_cfg['loop_cfg'], [])]
    general_cfg['experiment_name'] = '_'.join([c['keys'][-1] for c in relevant_cfg])
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
                           general_cfg['experiment_name'],
                           filename),
              )
