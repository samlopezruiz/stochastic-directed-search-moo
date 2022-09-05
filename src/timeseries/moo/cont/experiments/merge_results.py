import os

import joblib
from src.timeseries.utils.filename import get_result_folder
from src.timeseries.utils.files import save_vars

if __name__ == '__main__':
    # %%
    general_cfg = {'save_plot': False,
                   'save_results': False,
                   'show_title': False,
                   'experiment_name': 'model_ix',
                   'plot_individual_pf': False,
                   'color_per_subset': True,
                   }

    project = 'snp'

    files = [
        # ('eps', '2_ix_9_it'),
        # ('moo_batch_size', '2_ix_8_it_5'),
        # ('batch_gradient', '2_ix_2_it_1'),
        ('type_in_pf_eps_type_in_pf_eps_type_in_pf_eps', '2_ix_15_it_1'),
        ('type_in_pf_eps', '2_ix_1_it'),
    ]

    base_path = os.path.join(get_result_folder({}, project), 'experiments')
    # results_folder = os.path.join(project, 'compare', general_cfg['experiment_name'])
    cont_results = [joblib.load(os.path.join(get_result_folder({}, project), 'experiments', file[0], file[1]) + '.z')
                    for file in files]
    # cont_results = cont_results[0]
    # lbls = [', '.join(['{}:{}'.format(k, v) for k, v in d.items()]) for d in cont_results['exp_lbl']]
    # lbls = ['_'.join(['{}_{}'.format(k, v) for k, v in d.items()]) for d in cont_results['exp_lbl']]

    # %%

    params_cfg = cont_results[0]['params_cfg']

    lst = params_cfg['in_pf_eps_rank_s']['values']
    lst[3] = lst[4]
    lst[4] = cont_results[1]['params_cfg']['in_pf_eps_rank_s_one']['values'][0]
    params_cfg['in_pf_eps_rank_s']['values'] = lst

    # %%
    exp_lbl = cont_results[0]['exp_lbl']
    exp_lbl[8] = exp_lbl[9]
    exp_lbl[9] = cont_results[1]['exp_lbl'][0]

    # %%
    exp_results = cont_results[0]['exp_results']
    exp_results[8] = exp_results[9]
    exp_results[9] = cont_results[1]['exp_results'][0]

    # %%
    model_ix = 2
    filename = '2_ix_15_it_2'
    save_vars({'params_cfg': params_cfg,
               'exp_cfg': cont_results[0]['exp_cfg'],
               'exp_lbl': exp_lbl,
               'exp_results': exp_results},
              os.path.join(base_path,
                           'type_in_pf_eps_type_in_pf_eps_type_in_pf_eps',
                           filename),
              )
