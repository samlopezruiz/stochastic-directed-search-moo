from collections import defaultdict

import numpy as np
from numpy.linalg import norm

from src.timeseries.utils.moo import get_hypervolume, sort_arr_1st_col
from src.utils.plot import get_corrector_arrows, get_predictor_arrows


def norm_between_points(F):
    return [norm(F[i, :] - F[i - 1, :]) for i in range(1, F.shape[0])]


def metrics_of_pf(F, ref=[2., 2.]):
    if F is not None:
        dist_points = norm_between_points(F)
        return {'count': F.shape[0],
                'distances': dist_points,
                'hv': get_hypervolume(F, ref=ref),
                'mean norm': np.mean(dist_points),
                'std norm': np.std(dist_points),
                }
    else:
        return {}


def pred_corr_metrics(results):
    norms = defaultdict(list)
    descents = [res['descent'] for res in results['independent']]
    pops = [res['population'] for res in results['independent']]
    for i in range(len(descents)):
        pop, descent = pops[i], descents[i]
        c_points, c_uv, c_xy = get_corrector_arrows(pop['F_c'])
        d_points, d_uv, d_xy = get_corrector_arrows(descent['F'])
        p_points, p_uv, p_xy = get_predictor_arrows(pop['F_p'], pop['F'])

        norms['pred_norm'] += list(np.linalg.norm(p_uv, axis=1)) if len(p_uv) > 0 else []
        norms['corr_norm'] += list(np.linalg.norm(c_uv, axis=1)) if len(c_uv) > 0 else []
        norms['descent_norm'] += list(np.linalg.norm(d_uv, axis=1)) if len(d_uv) > 0 else []

        norms['n_correctors'] += [max(0, len(c) - 1) for c in pop['F_c']]
    norms['mean_correctors'] = np.mean(norms['n_correctors'])
    return norms


def subset_metrics(problem, results):
    metrics = defaultdict(dict)
    for subset in ['train', 'valid', 'test']:
        F = problem.eval_individuals(results['population']['X'], subset)
        if F is not None:
            F = sort_arr_1st_col(F)
            ini_fx = problem.eval_individuals([problem.original_x], subset)
            metrics[subset] = metrics_of_pf(F)
            metrics[subset]['F'] = F
            metrics[subset]['ini_fx'] = ini_fx

    return metrics
