import numpy as np

cont_cfg = {
    'model': {'basename': '60t_ema_q258',
              'results': 'TFTModel_ES_ema_r_q258_lr01_pred',
              'ix': 0},

    'data': {'shuffle': True,
             'random_state': 42},

    'problem': {'base_batch_size': 2 ** 11,
                'moo_batch_size': 1024,
                'quantile_ix': 0,
                'split_model': 'small',
                'limits': np.array([1., 1.])},

    'cont': {'step_eps': 2.5e-2,
             'verbose': True,
             'termination': {'type': 'none', 'thold': 8e-4},
             'single_descent': True,
             'max_increment': None,
             },

    'predictor': {'type': 'limit',
                  'eps': 1e-4,
                  },

    'corrector': {
        'type': 'rank',
        'in_pf_eps_cfg': {'small': {'delta': 0.00008, 'rank': 0.007, 'projection': 0.0009},
                          'medium': {'delta': 1e-4, 'rank': 0.007, 'projection': 5e-4}},
        # 'in_pf_eps': 1e-4,
        'batch_gradient': True,
        'mean_grad_stop_criteria': True,
        'batch_ratio_stop_criteria': 0.05,
        'step_eps': 2.5e-2,
        'maxiter': 30,
        't_fun': {'type': 'angle',
                  'eps': 110,
                  'maxiter': 50},
    },
}

params_cfg = {'steps': {'keys': ['cont', 'step_eps'],
                        'values': np.round(np.arange(0.005, 0.04, 0.01), 4)},

              'model': {'keys': ['model', 'results'],
                        'values': ['TFTModel_ES_ema_r_q258_lr01_pred']},

              'batch_size': {'keys': ['problem', 'moo_batch_size'],
                             'values': [2 ** i for i in range(7, 15)]},

              'model_ix': {'keys': ['model', 'ix'],
                           'values': list(range(1, 11))},

              'model_size': {'keys': ['problem', 'split_model'],
                             'values': ['small', 'medium']},

              'batch_gradient': {'keys': ['corrector', 'batch_gradient'],
                                 'values': [True, False]},

              'delta': {'keys': ['corrector', 'type'],
                        'values': ['delta']},

              'rank': {'keys': ['corrector', 'type'],
                       'values': ['rank']},

              'proj': {'keys': ['corrector', 'type'],
                       'values': ['projection']},

              'in_pf_eps_delta_l': {'keys': ['corrector', 'in_pf_eps'],
                                    'values': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]},

              'in_pf_eps_delta_s': {'keys': ['corrector', 'in_pf_eps'],
                                    'values': [5e-05, 0.0001, 0.0002, 0.0005, 0.001]},

              'in_pf_eps_rank_l': {'keys': ['corrector', 'in_pf_eps'],
                                   'values': [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]},

              'in_pf_eps_rank_s': {'keys': ['corrector', 'in_pf_eps'],
                                   'values': [0.005, 0.007, 0.009, 0.016, 0.02]},

              'in_pf_eps_rank_m': {'keys': ['corrector', 'in_pf_eps'],
                                   'values': [0.005, 0.007, 0.009]},

              'in_pf_eps_proj_l': {'keys': ['corrector', 'in_pf_eps'],
                                   'values': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]},

              'in_pf_eps_proj_s': {'keys': ['corrector', 'in_pf_eps'],
                                   'values': [0.0003, 0.0006, 0.0009, 0.0012, 0.0015]},

              't_fun_eps': {'keys': ['corrector', 't_fun', 'eps'],
                            'values': list(range(50, 140, 10))},

              'batch_sc_ratio': {'keys': ['corrector', 'batch_ratio_stop_criteria'],
                                 'values': [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]},
              }

experiments_cfg = {

    'model_ix': {'loop_cfg': {'model_ix': {}},
                 'ini_cfg': {'get_model': False,
                             'get_problem': False,
                             'get_cont': True,
                             'change_batch_size': False,
                             'seeds': np.arange(0, 10)}},

    'model': {'loop_cfg': {'model': {}},
              'ini_cfg': {'get_model': False,
                          'get_problem': False,
                          'get_cont': True,
                          'change_batch_size': False,
                          'seeds': np.arange(0, 10)}},

    'model_size': {'loop_cfg': {'model_size': {}},
                   'ini_cfg': {'get_model': False,
                               'get_problem': False,
                               'get_cont': True,
                               'change_batch_size': False,
                               'seeds': np.arange(0, 10)}},

    'batch_sc_ratio': {'loop_cfg': {'batch_sc_ratio': {}},
                       'ini_cfg': {'get_model': False,
                                   'get_problem': False,
                                   'get_cont': True,
                                   'change_batch_size': False,
                                   'seeds': np.arange(0, 10)}},

    'step_eps': {'loop_cfg': {'steps': {}},
                 'ini_cfg': {'get_model': False,
                             'get_problem': False,
                             'get_cont': True,
                             'change_batch_size': False,
                             'seeds': np.arange(0, 10)}},

    'batch_size': {'loop_cfg': {'batch_size': {}},
                   'ini_cfg': {'get_model': False,
                               'get_problem': False,
                               'get_cont': True,
                               'change_batch_size': True,
                               'seeds': np.arange(0, 10)}},

    'batch_gradient': {'loop_cfg': {'batch_gradient': {}},
                       'ini_cfg': {'get_model': False,
                                   'get_problem': False,
                                   'get_cont': True,
                                   'change_batch_size': False,
                                   'seeds': np.arange(0, 10)}},

    'in_pf_eps_m': {'loop_cfg': {'in_pf_eps_rank_m': {}},
                    'ini_cfg': {'get_model': False,
                                'get_problem': False,
                                'get_cont': True,
                                'change_batch_size': False,
                                'seeds': np.arange(0, 1)}},

    'condition': {'loop_cfg': {'delta': {'in_pf_eps_delta_s': {}},
                               'rank': {'in_pf_eps_rank_s': {}},
                               'proj': {'in_pf_eps_proj_s': {}}},
                  'ini_cfg': {'get_model': False,
                              'get_problem': False,
                              'get_cont': True,
                              'change_batch_size': False,
                              'seeds': np.arange(0, 10)}},

    'condition_large': {'loop_cfg': {'delta': {'in_pf_eps_delta_l': {}},
                                     'rank': {'in_pf_eps_rank_l': {}},
                                     'proj': {'in_pf_eps_proj_l': {}}},
                        'ini_cfg': {'get_model': False,
                                    'get_problem': False,
                                    'get_cont': True,
                                    'change_batch_size': False,
                                    'seeds': np.arange(0, 10)}},

    't_eps': {'loop_cfg': {'t_fun_eps': {}},
              'ini_cfg': {'get_model': False,
                          'get_problem': False,
                          'get_cont': True,
                          'change_batch_size': False,
                          'seeds': np.arange(0, 10)}}

}
