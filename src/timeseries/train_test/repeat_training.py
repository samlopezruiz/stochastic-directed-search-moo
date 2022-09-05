import os
import time

import pandas as pd
import tensorflow as tf
import telegram_send
from src.timeseries.plot.ts import plotly_time_series
from src.timeseries.utils.config import read_config
from src.timeseries.utils.filename import quantiles_name
from src.timeseries.utils.files import save_vars
from src.timeseries.utils.harness import get_model_data_config, train_test_model
from src.timeseries.utils.results import post_process_results

if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Device Name: ", tf.test.gpu_device_name())
    print('TF eager execution: {}'.format(tf.executing_eagerly()))

general_cfg = {'send_notifications': True,
               'save_results': True, }

project = 'snp'
experiment_cfg = {'experiment_name': '60t_ema_q258_0',
                  'model_cfg': 'q258_i48_o5_h4_e100',
                  'preprocess_cfg': 'ES_60t_regime_2015_1_to_2021_6_grp_w8_ema_r',
                  'vars_definition_cfg': 'ES_ema_r',
                  'architecture': 'TFTModel'
                  }

n_repeat = 11
ini = 4
for i in range(ini, ini + n_repeat):
    experiment_cfg['experiment_name'] = experiment_cfg['experiment_name'][:-1] + str(i)
    model_cfg = read_config(experiment_cfg['model_cfg'], project, subfolder='model')
    config, data_formatter, model_folder = get_model_data_config(project,
                                                                 experiment_cfg,
                                                                 model_cfg['model_params'],
                                                                 model_cfg['fixed_params'])
    t0 = time.time()
    results = train_test_model(use_gpu=True,
                               architecture=experiment_cfg['architecture'],
                               prefetch_data=False,
                               model_folder=model_folder,
                               data_config=config.data_config,
                               data_formatter=data_formatter,
                               use_testing_mode=False,
                               predict_eval=True,
                               tb_callback=False,
                               use_best_params=False,
                               indicators_use_time_subset=True
                               )

    filename = '{}_{}_q{}_lr{}_pred'.format(experiment_cfg['architecture'],
                                            experiment_cfg['vars_definition_cfg'],
                                            quantiles_name(results['quantiles']),
                                            str(results['learning_rate'])[2:],
                                            )

    if general_cfg['send_notifications']:
        try:
            mins = round((time.time() - t0) / 60, 0)
            gens = 'in {} epochs'.format(
                len(results['fit_history']['loss']) if results['fit_history'] is not None else '')
            telegram_send.send(messages=["training for {} completed in {} mins {}".format(filename, mins, gens)])
        except Exception as e:
            print(e)

    post_process_results(results, data_formatter, experiment_cfg)

    if general_cfg['save_results']:
        results['model_params'] = model_cfg['model_params']
        results['fixed_params'] = model_cfg['fixed_params']
        results['experiment_cfg'] = experiment_cfg
        save_vars(results, os.path.join(config.results_folder,
                                        experiment_cfg['experiment_name'],
                                        filename))

# %%
# histories = [res['fit_history']['loss'] for res in results]
#
# df = pd.DataFrame(histories).T
# plotly_time_series(df,
#                    title='Loss History',
#                    save=False,
#                    legend=True,
#                    rows=[1, 1],
#                    file_path=os.path.join(config.results_folder,
#                                           'img',
#                                           '{}_{}_loss_hist'.format(experiment_cfg['architecture'],
#                                                                    experiment_cfg['vars_definition_cfg'])),
#                    size=(1980, 1080),
#                    color_col=None,
#                    markers='lines+markers',
#                    xaxis_title="epoch",
#                    markersize=5,
#                    plot_title=True,
#                    label_scale=1,
#                    plot_ytitles=False)

# post_process_results(results, formatter, experiment_cfg)
#
# if general_cfg['save_forecast']:
#     save_vars(results, os.path.join(config.results_folder,
#                                     experiment_cfg['experiment_name'],
#                                     '{}_{}_forecasts'.format(experiment_cfg['architecture'],
#                                                              experiment_cfg['vars_definition'])))
#
# print(results['hit_rates']['global_hit_rate'][1])
