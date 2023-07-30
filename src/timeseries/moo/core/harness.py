import os
import time
from collections import defaultdict
from copy import deepcopy

import joblib
import numpy as np
from tabulate import tabulate

from src.sds.core.continuation import BiDirectionalDsContinuation
from src.sds.factory import get_corrector, get_predictor, get_tfun, get_cont_termination
from src.sds.utils.util import subroutine_times_problem
from src.timeseries.moo.core.problem import TsQuantileProblem
from src.timeseries.moo.sds.utils.indicators import subset_metrics, pred_corr_metrics
from src.timeseries.moo.sds.utils.results import compile_metrics_from_results, concat_mean_std_cols
from src.timeseries.moo.sds.utils.util import set_in_dict, get_from_dict
from src.timeseries.utils.continuation import get_q_moo_params_for_problem2
from src.timeseries.utils.filename import get_result_folder
from src.timeseries.utils.files import save_vars
from src.timeseries.utils.moo import get_hypervolume, sort_1st_col
from src.timeseries.utils.util import write_text_file, latex_table
from src.utils.plot import plot_bidir_2D_points_vectors, plot_2D_points_traces_total, get_corrector_arrows, \
    get_predictor_arrows
import tensorflow as tf


def get_model_and_params(cont_cfg, project, use_gpu=True):
    cont_cfg['model']['experiment_name'] = cont_cfg['model']['basename'] + '_' + str(cont_cfg['model']['ix'])
    results_folder = get_result_folder(cont_cfg['model'], project)
    model_results = joblib.load(os.path.join(results_folder, cont_cfg['model']['results'] + '.z'))
    model_params = get_q_moo_params_for_problem2(project,
                                                 model_results,
                                                 shuffle_data=cont_cfg['data']['shuffle'],
                                                 random_state=cont_cfg['data']['random_state'],
                                                 use_gpu=use_gpu)
    return model_params, results_folder


def run_cont_problem(ds_cont, problem):
    # Run problem
    t0 = time.time()
    problem.n_f_evals, problem.n_grad_evals = 0, 0
    results = ds_cont.run(np.reshape(problem.original_x, (-1)))
    exec_time = round(time.time() - t0, 2)
    print('time: {} s'.format(exec_time))
    # print('f(x) evals: {}, dx(x) evals: {:.2}'.format(problem.n_f_evals, problem.n_grad_evals))

    print(tabulate([[name, *inner.values()] for name, inner in results['evaluations'].items()],
                   tablefmt='psql',
                   headers=list(results['evaluations'][list(results['evaluations'].keys())[0]].keys())))

    print('post-processing results...', end='')
    results['exec_time'] = exec_time
    results['pred_corr_metrics'] = pred_corr_metrics(results)
    results['subset_metrics'] = subset_metrics(problem, results)

    metrics = compile_metrics_from_results(results, problem)
    print('Done!')
    return results, metrics


def get_ts_problem(cont_cfg, model_params, test_ss=True, use_gpu=True):
    t0 = time.time()
    problem = TsQuantileProblem(y_train=model_params['datasets']['train']['y'],
                                x_train=model_params['datasets']['train']['x'],
                                y_valid=model_params['datasets']['valid']['y'],
                                x_valid=model_params['datasets']['valid']['x'],
                                y_test=model_params['datasets']['test']['y'] if test_ss else None,
                                x_test=model_params['datasets']['test']['x'] if test_ss else None,
                                model=model_params['model'].model,
                                eval_fs=[model_params['loss']],
                                constraints_limits=cont_cfg['problem']['limits'],
                                quantile_ix=cont_cfg['problem']['quantile_ix'],
                                base_batch_size=cont_cfg['problem']['base_batch_size'],
                                moo_batch_size=cont_cfg['problem']['moo_batch_size'],
                                moo_model_size=cont_cfg['problem']['split_model'],
                                use_gpu=use_gpu)

    print('init core time: {}'.format(round(time.time() - t0, 4)))
    times = subroutine_times_problem(problem)
    print(tabulate(times, headers='keys', floatfmt=(None, ",.4f"), tablefmt='psql'))

    return problem


def get_continuation_method(cont_cfg, problem):
    problem.constraints_limits = None
    problem.n_constr = 0
    predictor = get_predictor(cont_cfg['predictor']['type'],
                              eps=cont_cfg['predictor']['eps'],
                              problem=problem,
                              limits=cont_cfg['problem']['limits'],
                              max_increment=cont_cfg['sds']['max_increment'])

    in_pf_eps = cont_cfg['corrector']['in_pf_eps'] if 'in_pf_eps' in cont_cfg['corrector'] else \
        cont_cfg['corrector']['in_pf_eps_cfg'][cont_cfg['problem']['split_model']][cont_cfg['corrector']['type']]
    corrector = get_corrector(cont_cfg['corrector']['type'],
                              problem=problem,
                              t_fun=get_tfun(cont_cfg['corrector']['t_fun']['type'],
                                             problem=problem,
                                             eps=cont_cfg['corrector']['t_fun']['eps'],
                                             maxiter=cont_cfg['corrector']['t_fun']['maxiter']),
                              a_fun=lambda a, dx: a,
                              batch_gradient=cont_cfg['corrector']['batch_gradient'],
                              mean_grad_stop_criteria=cont_cfg['corrector']['mean_grad_stop_criteria'],
                              batch_ratio_stop_criteria=cont_cfg['corrector']['batch_ratio_stop_criteria'],
                              step_eps=cont_cfg['corrector']['step_eps'],
                              in_pf_eps=in_pf_eps,
                              maxiter=cont_cfg['corrector']['maxiter']
                              )

    ds_cont = BiDirectionalDsContinuation(problem,
                                          predictor,
                                          corrector,
                                          get_cont_termination(cont_cfg['sds']['termination']['type'],
                                                               cont_cfg['sds']['termination']['thold']),
                                          limits=cont_cfg['problem']['limits'],
                                          step_eps=cont_cfg['sds']['step_eps'],
                                          verbose=cont_cfg['sds']['verbose'],
                                          single_descent=cont_cfg['sds']['single_descent'],
                                          # max_increment=cont_cfg['sds']['max_increment'],
                                          )
    return ds_cont


def filename_from_cfg(cont_cfg):
    filename = 'm_{}_sz_{}_b_{}_bs_{}_st_{}_ct_{}_t_{}'.format(cont_cfg['model']['ix'],
                                                               cont_cfg['problem']['split_model'],
                                                               cont_cfg['corrector']['batch_gradient'],
                                                               cont_cfg['problem']['moo_batch_size'],
                                                               str(cont_cfg['sds']['step_eps'])[2:],
                                                               cont_cfg['corrector']['type'],
                                                               str(cont_cfg['corrector']['in_pf_eps_cfg'][
                                                                       cont_cfg['problem']['split_model']][
                                                                       cont_cfg['corrector']['type']])[2:],
                                                               )
    return filename


def title_from_cfg(cont_cfg):
    filename = 'model: {}, size: {}, batch: {}-{}, step: {}, criteria: {}, tol: {}'.format(cont_cfg['model']['ix'],
                                                                                           cont_cfg['problem'][
                                                                                               'split_model'],
                                                                                           cont_cfg['corrector'][
                                                                                               'batch_gradient'],
                                                                                           cont_cfg['problem'][
                                                                                               'moo_batch_size'],
                                                                                           cont_cfg['sds'][
                                                                                               'step_eps'],
                                                                                           cont_cfg['corrector'][
                                                                                               'type'],
                                                                                           cont_cfg['corrector'][
                                                                                               'in_pf_eps_cfg'][
                                                                                               cont_cfg['problem'][
                                                                                                   'split_model']][
                                                                                               cont_cfg[
                                                                                                   'corrector'][
                                                                                                   'type']],
                                                                                           )
    return filename


def plot_pf_and_total(results, results_folder, cfg, cont_cfg):
    filename = filename_from_cfg(cont_cfg)
    img_path = os.path.join(results_folder, 'sds', 'img', filename)
    train_population = [res['population'] for res in results['independent']]
    descent_pops = [res['descent'] for res in results['independent']]

    title = title_from_cfg(cont_cfg)
    plot_bidir_2D_points_vectors(train_population,
                                 pareto=None,
                                 descent=descent_pops,
                                 arrow_scale=0.4,
                                 markersize=5,
                                 pareto_marker_mode='markers+lines',
                                 save=cfg['save_plots'],
                                 save_png=False,
                                 file_path=img_path + '_pf',
                                 size=(1980, 1080),
                                 plot_arrows=True,
                                 plot_points=True,
                                 plot_ps=False,
                                 plot_title=cfg['plot_title'],
                                 return_fig=False,
                                 titles=[title, title] if cfg['plot_title'] else [None, None])

    X_sorted, F_sorted = sort_1st_col(results['population']['X'], results['population']['F'])
    fx_ini = results['independent'][0]['descent']['ini_fx'].reshape(1, 2)

    plot_2D_points_traces_total([F_sorted, fx_ini],
                                names=['sds', 'ini'],
                                markersizes=[10, 20],
                                marker_symbols=['circle', 'asterisk'],
                                modes=['markers+lines', 'markers'],
                                save=cfg['save_plots'],
                                outlines=[False, True],
                                label_scale=1.8,
                                file_path=img_path + '_total',
                                axes_labels=('Quantile coverage risk', 'Quantile estimation risk'))

    # subset plots
    Fs = [s['F'] for _, s in results['subset_metrics'].items()]
    fx_inis = [s['ini_fx'] for _, s in results['subset_metrics'].items()]

    names = ['PF_' + str(i) for i in results['subset_metrics'].keys()] + ['ini_fx_' + str(i) for i in
                                                                          results['subset_metrics'].keys()]

    plot_2D_pf(Fs, fx_inis, names, cfg['save_plots'], img_path, f_markersize=10, label_scale=1.8)


def plot_2D_pf(Fs,
               fx_inis,
               names,
               save,
               img_path,
               f_markersize=5,
               f_mode='markers+lines',
               colors_ixs=None,
               axes_labels=('Quantile coverage risk', 'Quantile estimation risk'),
               **kwargs):

    data = Fs + fx_inis
    modes = [f_mode] * len(Fs) + ['markers'] * len(fx_inis)
    colors_ixs = list(range(len(Fs))) * 2 if colors_ixs is None else colors_ixs
    markersizes = [f_markersize] * len(Fs) + [10] * len(fx_inis)
    marker_symbols = ['circle'] * len(Fs) + ['hexagram'] * len(fx_inis)
    outlines = [False] * len(Fs) + [True] * len(fx_inis)
    show_legends = [True] * len(Fs) + [False] * len(fx_inis)
    plot_2D_points_traces_total(data,
                                names,
                                markersizes,
                                colors_ixs,
                                modes,
                                marker_symbols,
                                outlines,
                                show_legends=show_legends,
                                axes_labels=axes_labels,
                                save=save,
                                file_path=img_path + '_subsets',
                                **kwargs)


def save_cont_resuls(results, results_folder, cfg, cont_cfg):
    filename = filename_from_cfg(cont_cfg)
    res_path = os.path.join(results_folder, 'sds', filename)
    if cfg['save_results']:
        save_vars(results, res_path)


def save_latex_table(metrics, results_folder, cfg, cont_cfg):
    if cfg['save_latex']:
        filename = filename_from_cfg(cont_cfg)

        for key, title in zip(['pred_corr_metrics', 'subset_metrics', 'times'],
                              ['Prediction and correction metrics', 'Subset metrics', 'Execution times']):

            df = metrics[key].copy()
            if 'mean norm' in metrics[key].columns:
                df = concat_mean_std_cols(df, 'mean norm', 'std norm', 'norm', round_dec=4)
            if 'mean per step' in metrics[key].columns:
                df = concat_mean_std_cols(df, 'mean per step', 'std per step', 'per step', round_dec=4)

            df.columns = [c.replace('_', ' ') for c in df.columns]
            print(tabulate(df, headers='keys', tablefmt='psql'))
            write_text_file(os.path.join(results_folder,
                                         'sds',
                                         'txt',
                                         '{}_{}'.format(filename, key)),
                            latex_table(title, df.to_latex(escape=False)))


def get_cfg_from_loop_cfg(loop_cfg, params_cfg, cont_cfg, cfgs=[]):
    if len(loop_cfg.keys()) > 0:
        for key in loop_cfg.keys():
            for value in params_cfg[key]['values']:
                new_cfg = deepcopy(cont_cfg)
                set_in_dict(new_cfg, params_cfg[key]['keys'], value)
                get_cfg_from_loop_cfg(loop_cfg[key], params_cfg, new_cfg, cfgs)
    else:
        cfgs.append(cont_cfg)

    return cfgs


def get_sublevel_keys(d, keys=[]):
    for k, v in d.items():
        keys.append(k)
        get_sublevel_keys(d[k], keys)
    return keys


def run_experiments(cfgs,
                    project,
                    relevant_cfg,
                    get_model,
                    get_problem,
                    get_cont,
                    change_batch_size,
                    use_gpu=True,
                    seeds=None,
                    ):
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

        if change_batch_size and i > 0:
            problem.moo_batch_size = get_from_dict(cfg, ['problem', 'moo_batch_size'])
            problem.batch_moo_inputs()

        if (get_model or get_problem) and i > 0:
            tf.keras.backend.clear_session()
            problem = get_ts_problem(cfg, model_params, test_ss=False, use_gpu=use_gpu)

        if (get_model or get_cont) and i > 0:
            ds_cont = get_continuation_method(cfg, problem)

        if seeds is not None:
            results, metrics = [], []
            for i, seed in enumerate(seeds):
                print('{}/{}: shuffling data with seed: {}'.format(i + 1, len(seeds), seed))
                problem.shuffle_train_data(random_state=seed)
                # reset batch ix
                problem.train_batch_ix = 0
                res, met = run_cont_problem(ds_cont, problem)
                results.append(res)
                metrics.append(met)
                ds_cont.reset()
        else:
            problem.train_batch_ix = 0
            results, metrics = run_cont_problem(ds_cont, problem)

        exp_results.append({'results': results, 'metrics': metrics})

    return exp_results, results_folder
