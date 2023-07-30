from src.timeseries.moo.sds.config import sds_cfg
from src.timeseries.moo.core.harness import get_model_and_params, get_ts_problem, get_continuation_method, \
    run_cont_problem, save_cont_resuls, save_latex_table, plot_pf_and_total
from src.timeseries.moo.sds.utils.bash import get_input_args
from src.timeseries.moo.sds.utils.util import get_from_dict

if __name__ == '__main__':
    # %%
    input_args = get_input_args()

    cfg = {'save_plots': False,
           'save_results': True,
           'save_latex': True,
           'plot_title': False,
           }

    project = 'snp'

    sds_cfg['model']['ix'] = input_args['model_ix']
    sds_cfg['model']['ix'] = 0
    print('Model ix: {}'.format(get_from_dict(sds_cfg, ['model', 'ix'])))

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

