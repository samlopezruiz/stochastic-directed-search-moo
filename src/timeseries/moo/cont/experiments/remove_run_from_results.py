import os

import joblib

from src.timeseries.utils.filename import get_result_folder
from src.timeseries.utils.files import save_vars

if __name__ == '__main__':
    # %%

    project = 'snp'

    file = ('model_ix', '10_it_5')
    remove_ix = 8
    file_name = '10_it_6'

    base_path = os.path.join(get_result_folder({}, project), 'experiments')
    cont_results = joblib.load(os.path.join(get_result_folder({}, project), 'experiments', file[0], file[1]) + '.z')
    # %%
    cont_results['exp_results'] = [x for i, x in enumerate(cont_results['exp_results']) if i != remove_ix]
    cont_results['exp_lbl'] = cont_results['exp_lbl'][:-1]
    # cont_results['exp_lbl'] = [x for i, x in enumerate(cont_results['exp_lbl']) if i != remove_ix]

    save_vars(cont_results,
              os.path.join(base_path,
                           file[0],
                           file_name),
              )
