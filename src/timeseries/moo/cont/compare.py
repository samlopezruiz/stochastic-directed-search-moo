import os
import time

import joblib
import numpy as np
from numpy.linalg import pinv
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination, get_reference_directions
from pymoo.optimize import minimize
from tabulate import tabulate

from src.models.attn.hyperparam_opt import HyperparamOptManager
from src.moo.core.continuation import BiDirectionalDsContinuation
from src.moo.factory import get_corrector, get_predictor, get_tfun, get_cont_termination
from src.timeseries.moo.cont.core.problem import TsQuantileProblem
from src.timeseries.utils.continuation import get_q_moo_params_for_problem
from src.timeseries.utils.filename import get_result_folder
from src.timeseries.utils.files import save_vars
from src.timeseries.utils.harness import get_model_data_config, get_model
from src.timeseries.utils.moo import get_hypervolume, sort_1st_col
from src.utils.plot import plot_bidir_2D_points_vectors, plot_2D_points_traces, plot_2D_points_traces_total

if __name__ == '__main__':
    # %%
    cfg = {'save_plots': False,
           'problem_name': 'ts_q',
           'split_model': "small",
           'quantile_ix': 0,
           }

    solve_moea = False
    project = 'snp'
    results_cfg = [{'experiment_name': '60t_ema_q258_0', 'results': 'delta'},
                   {'experiment_name': '60t_ema_q258_1', 'results': 'delta'},
                   {'experiment_name': '60t_ema_q258_2', 'results': 'delta'},
                   {'experiment_name': '60t_ema_q258_3', 'results': 'delta'},
                   {'experiment_name': '60t_ema_q258_4', 'results': 'delta'},
                   {'experiment_name': '60t_ema_q258_5', 'results': 'delta'}]

    cont_results = [joblib.load(os.path.join(get_result_folder(cfg, project), 'cont',
                                             cfg['results'] + '.z')) for cfg in results_cfg]

    # %%
    Fs, fx_inis = [], []
    for results in cont_results:
        X_sorted, F_sorted = sort_1st_col(results['population']['X'], results['population']['F'])
        Fs.append(F_sorted)
        fx_inis.append(results['independent'][0]['descent']['ini_fx'].reshape(1, 2))

    data = Fs + fx_inis
    names = ['F_' + str(i) for i in range(len(Fs))] + ['ini_fx_' + str(i) for i in range(len(fx_inis))]
    modes = ['markers+lines'] * len(Fs) + ['markers'] * len(fx_inis)
    colors_ixs = list(range(len(Fs))) * 2
    markersizes = [5] * len(Fs) + [15] * len(fx_inis)
    marker_symbols = ['circle'] * len(Fs) + ['hexagram'] * len(fx_inis)
    outlines = [False] * len(Fs) + [True] * len(fx_inis)
    plot_2D_points_traces_total(data, names, markersizes, colors_ixs, modes, marker_symbols, outlines,
                                save=True, save_png=True,
                                file_path=os.path.join(get_result_folder(cfg, project), 'compare',
                                                       'PF_different_models'))
