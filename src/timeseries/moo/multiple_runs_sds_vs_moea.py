import time

import joblib
import pandas as pd
from pymoo.core.callback import Callback
from tabulate import tabulate

from src.models.compare.winners import wilcoxon_significance
from src.timeseries.moo.sds.config import sds_cfg
from src.timeseries.moo.core.harness import get_model_and_params, get_ts_problem, get_continuation_method, \
    run_cont_problem
from src.timeseries.moo.sds.utils.bash import get_input_args
from src.timeseries.moo.sds.utils.indicators import metrics_of_pf
from src.timeseries.moo.sds.utils.util import get_from_dict
from src.timeseries.utils.moo import sort_1st_col
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination, get_reference_directions
from pymoo.optimize import minimize
import numpy as np


class MetricsCallback(Callback):

    def __init__(self, gens_to_measure, problem, t0) -> None:
        super().__init__()
        self.gens_to_measure = gens_to_measure
        self.problem = problem
        self.t0 = t0
        self.data["metrics"] = []
        self.data["times"] = []

    def notify(self, algorithm, **kwargs):
        if algorithm.n_gen in self.gens_to_measure:
            X = algorithm.pop.get("X")
            F = problem.eval_individuals(X, 'valid')
            X_moea_sorted, F_moea_sorted = sort_1st_col(X, F)
            moea_metrics = metrics_of_pf(F_moea_sorted, ref=[2., 2.])
            self.data["metrics"].append(moea_metrics)
            self.data["times"].append(time.time() - t0)


if __name__ == '__main__':
    # %%
    input_args = get_input_args()

    cfg = {'save_plots': False,
           'save_results': False,
           'save_latex': False,
           'plot_title': False,
           }

    project = 'snp'

    sds_cfg['model']['ix'] = input_args['model_ix']
    sds_cfg['model']['ix'] = 0
    sds_cfg['problem']['split_model'] = 'small'
    print('Model ix: {}'.format(get_from_dict(sds_cfg, ['model', 'ix'])))

    model_params, results_folder = get_model_and_params(sds_cfg, project)
    problem = get_ts_problem(sds_cfg, model_params, test_ss=False)
    ds_cont = get_continuation_method(sds_cfg, problem)
    n_repeat = 3

    #%% Solve N times with SDS
    sds_results, sds_metrics = [], []
    seeds = range(n_repeat)
    for i, seed in enumerate(seeds):
        print('{}/{}: shuffling data with seed: {}'.format(i + 1, len(seeds), seed))
        problem.shuffle_train_data(random_state=seed)
        # reset batch ix
        problem.train_batch_ix = 0
        res, met = run_cont_problem(ds_cont, problem)
        sds_results.append(res)
        sds_metrics.append(met)
        joblib.dump(met, f'../output/sds_results{seed}.pkl')
        joblib.dump(res, f'../output/sds_full_results{seed}.pkl')
        ds_cont.reset()

    # %%
    rhos = np.array([r['times'].loc['J(x)', 'mean (s)'] / r['times'].loc['f(x)', 'mean (s)']  for r in sds_metrics])
    fevals = np.array([r['pred_corr_metrics']['f_evals'].sum() for r in sds_metrics])
    Jevals = np.array([r['pred_corr_metrics']['grad_evals'].sum() for r in sds_metrics])
    evals = Jevals * rhos + fevals
    print(np.mean(evals), np.std(evals))

    times = [r['times'].loc['execution', 'mean (s)'] for r in sds_metrics]
    print(np.mean(times), np.std(times))

    print(np.mean([r['subset_metrics'].loc['valid', 'mean norm'] for r in sds_metrics]), np.std([r['subset_metrics'].loc['valid', 'std norm'] for r in sds_metrics]))

    sds_hvs = [r['subset_metrics'].loc['valid', 'hv'] for r in sds_metrics]
    print(np.mean(sds_hvs), np.std(sds_hvs))

    # %% Solve N times with MOEA (take measurements along generations)
    # problem.constraints_limits = [0.459, .583]
    problem.constraints_limits = [1.0, 1.0]
    problem.n_constr = 2
    pop_size, n_gen = 60, 50
    gens_to_measure = [14, 25, n_gen] # 50, 75, 100, 125, n_gen]

    moea_results = []
    times = []
    for seed in range(n_repeat):
        t0 = time.time()
        algorithm = NSGA2(
            pop_size=pop_size,
            n_offsprings=pop_size,
            ref_dirs=get_reference_directions("energy", problem.n_obj, pop_size),
            sampling=get_sampling("real_random"),
            crossover=get_crossover("real_sbx", prob=0.9, eta=15),
            mutation=get_mutation("real_pm", eta=20),
            eliminate_duplicates=True
        )
        termination = get_termination("n_gen", n_gen)
        res = minimize(problem,
                       algorithm,
                       termination,
                       seed=seed,
                       save_history=False,
                       verbose=True,
                       callback=MetricsCallback(gens_to_measure, problem, t0))

        moea_results.append(res.algorithm.callback.data["metrics"])
        times.append(res.algorithm.callback.data["times"])

        joblib.dump((res.algorithm.callback.data["metrics"], res.algorithm.callback.data["times"]), f'../output/moea_results_{seed}.pkl')

    #%%
    gens_res = []
    for c, g in enumerate(gens_to_measure):
        ts = [t[c] for t in times]

        distances = [r[c]['distances'] for r in moea_results]
        distances = [item for listoflists in distances for item in listoflists]
        hvs = [r[c]['hv'] for r in moea_results]

        gens_res.append({'generation': g,
                         'time': '{:.2f} ({:.2f})'.format(np.mean(ts), np.std(ts)),
                         'distance': '{:.4f} ({:.4f})'.format(np.mean(distances), np.std(distances)),
                         'hv': '{:.4f} ({:.4f})'.format(np.mean(hvs), np.std(hvs))})

    gens_res_df = pd.DataFrame.from_records(gens_res)
    print(gens_res_df)

    ws = wilcoxon_significance([sds_hvs, hvs], ['SDS', 'NSGA-II'])
    print(tabulate(ws, headers='keys', tablefmt='psql'))

