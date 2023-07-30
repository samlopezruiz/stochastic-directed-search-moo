import os

import joblib
import plotly.io as pio

from src.timeseries.utils.filename import get_result_folder
from src.timeseries.utils.files import save_vars

pio.renderers.default = "browser"

if __name__ == '__main__':
    # %%
    project = 'snp'

    files = [
        ('split_model', '2_ix_2_it'),
    ]

    cont_results = [joblib.load(os.path.join(get_result_folder({}, project), 'experiments', file[0], file[1]) + '.z')
                    for file in files]
    cont_results = cont_results[0]

    # %%#%%
    for result in cont_results['exp_results']:
        for res in result['results']:
            for ind in res['independent']:
                del ind['population']['X_p']
                del ind['population']['X_c']
                del ind['population']['X']
                del ind['population']['vs']
                del ind['descent']['X']
                del ind['descent']['ini_x']

            del res['population']['X']
            del res['population']['X_r']
            del res['population']['vs']

    base_path = os.path.join(get_result_folder({}, project), 'experiments')

    filename = '2_ix_2_it_woX3'
    save_vars(cont_results,
              os.path.join(base_path,
                           'split_model',
                           filename),
              )
