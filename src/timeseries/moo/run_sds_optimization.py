import os

import joblib

from src.timeseries.moo.sds.config import sds_cfg
from src.timeseries.moo.core.harness import get_model_and_params, get_ts_problem, get_continuation_method, \
    run_cont_problem, save_cont_resuls, save_latex_table, plot_pf_and_total
from src.timeseries.moo.sds.utils.bash import get_input_args
from src.timeseries.moo.sds.utils.util import get_from_dict, set_in_dict
from src.timeseries.utils.filename import get_result_folder
from src.timeseries.utils.files import save_vars

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
    print('Model ix: {}'.format(get_from_dict(sds_cfg, ['model', 'ix'])))

    #%%
    # sds_cfg['model']['experiment_name'] = sds_cfg['model']['basename'] + '_' + str(sds_cfg['model']['ix'])
    # results_folder = get_result_folder(sds_cfg['model'], project)
    # model_results = joblib.load(os.path.join(results_folder, sds_cfg['model']['results'] + '.z'))
    # model_results['experiment_cfg']['experiment_name'] = sds_cfg['model']['experiment_name']
    # save_vars(model_results, os.path.join(results_folder, sds_cfg['model']['results']))
    #%%
    model_params, results_folder = get_model_and_params(sds_cfg, project)
    problem = get_ts_problem(sds_cfg, model_params, test_ss=False)
    ds_cont = get_continuation_method(sds_cfg, problem)

    #%% Optimize with SDS
    results, metrics = run_cont_problem(ds_cont, problem)

    # Save results
    save_cont_resuls({'results': results, 'metrics': metrics, 'cont_cfg': sds_cfg}, results_folder, cfg, sds_cfg)

    # Save latex tables
    save_latex_table(metrics, results_folder, cfg, sds_cfg)

    # Plot results
    plot_pf_and_total(results, results_folder, cfg, sds_cfg)

