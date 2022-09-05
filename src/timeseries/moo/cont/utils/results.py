from collections import defaultdict

import numpy as np
import pandas as pd

from src.moo.utils.util import subroutine_times_problem


def concat_mean_std_cols(df, mean_col, std_col, new_col, round_dec=4):
    new_df = df.copy()
    new_df = new_df.applymap(lambda x: round(x, round_dec) if isinstance(x, float) else x)
    new_df[new_col] = new_df[mean_col].astype(str) + ' $\\pm$ (' + new_df[std_col].astype(str) + ')'
    new_df.drop(mean_col, axis=1, inplace=True)
    new_df.drop(std_col, axis=1, inplace=True)
    return new_df


def mean_std_count_from_list(lst):
    return np.mean(lst), np.std(lst), len(lst)


def compile_metrics_from_results(results, problem):
    pred_corr_metrics = pd.DataFrame()
    pred_corr_metrics['predictor'] = mean_std_count_from_list(results['pred_corr_metrics']['pred_norm'])
    pred_corr_metrics['corrector'] = mean_std_count_from_list(results['pred_corr_metrics']['corr_norm'])
    pred_corr_metrics['descent'] = mean_std_count_from_list(results['pred_corr_metrics']['descent_norm'])
    pred_corr_metrics.index = ['mean norm', 'std norm', 'count']
    pred_corr_metrics = pred_corr_metrics.T
    pred_corr_metrics['mean per step'] = [1, round(pred_corr_metrics.loc['corrector', 'count'] / pred_corr_metrics.loc[
        'predictor', 'count'], 4), '-']
    pred_corr_metrics['std per step'] = [0, round(np.std(results['pred_corr_metrics']['n_correctors']), 4), '-']
    pred_corr_metrics['f_evals'] = [results['evaluations']['f'][e] for e in ['predictor', 'corrector', 'descent']]
    pred_corr_metrics['grad_evals'] = [results['evaluations']['grad'][e] for e in ['predictor', 'corrector', 'descent']]

    # Subset metrics
    keys = list(results['subset_metrics']['train'].keys())
    [keys.remove(k) for k in ['distances', 'F', 'ini_fx']]

    subset_metrics = pd.DataFrame()
    for subset in results['subset_metrics'].keys():
        subset_metrics[subset] = [results['subset_metrics'][subset][k] for k in keys]
    subset_metrics.index = keys
    subset_metrics = subset_metrics.T
    times = subroutine_times_problem(problem, results)

    return {'pred_corr_metrics': pred_corr_metrics, 'subset_metrics': subset_metrics, 'times': times}


def adapt_runs(compiled_dict):
    new_dict = {'lbls': compiled_dict['lbls']}
    for key, value in compiled_dict.items():
        if key != 'lbls':
            if 'mean' in value.keys() and 'std' in value.keys() and 'count' in value.keys():
                if len(value['mean'].shape) > 1:
                    muc, stc = combine_means_stds(value['mean'], value['std'], value['count'])
                    new_dict[key] = {'mean': muc, 'std': stc}
                else:
                    new_dict[key] = {'mean': value['mean'], 'std': value['std']}
            elif 'mean' in value.keys() and 'std' not in value.keys():
                if len(value['mean'].shape) > 1:
                    new_dict[key] = {'mean': np.mean(value['mean'], axis=1), 'std': np.std(value['mean'], axis=1)}
                else:
                    new_dict[key] = value
            else:
                new_dict[key] = value
    return new_dict


def combine_means_stds(mu, st, cnt):
    """
    :param mu: shape: (n experiments, n distributions)
    :param st:
    :param cnt:
    :return:
    """
    muc = np.sum(np.multiply(mu, cnt), axis=1) / np.sum(cnt, axis=1)
    qc = (cnt - 1) * (st ** 2) + cnt * (mu ** 2)
    qc = np.sum(qc, axis=1)
    stc = np.sqrt((np.sum((cnt - 1) * (st ** 2), axis=1) / (np.sum(cnt, axis=1) + cnt.shape[1])))

    return muc, stc


def df_from_dict(d):
    df = pd.DataFrame()
    for base_name, values in d.items():
        if base_name in ['lbls']:
            continue

        if 'std' in values:
            df[base_name] = values['mean']
            df[base_name + '_std'] = values['std']
        else:
            df[base_name] = values['mean']
    df.index = d['lbls']
    return df


def get_compiled_dict(dfs, lbls, get_values):
    plot_vals = defaultdict(dict)
    plot_vals['lbls'] = lbls
    for key, data_cfg in get_values.items():
        for name, rowcol in data_cfg.items():
            if isinstance(dfs[0], list):
                plot_vals[key][name] = np.array([[df.loc[rowcol[0], rowcol[1]] for df in df_lst] for df_lst in dfs])
            else:
                plot_vals[key][name] = np.array([df.loc[rowcol[0], rowcol[1]] for df in dfs])
    return plot_vals


def compile_metrics(lbls, dfs_list, get_values_list):
    assert len(dfs_list) == len(get_values_list)

    compiled_dict = {}
    for i in range(len(dfs_list)):
        d = get_compiled_dict(dfs_list[i], lbls, get_values_list[i])
        compiled_dict.update(adapt_runs(d))

    return compiled_dict, df_from_dict(compiled_dict)
