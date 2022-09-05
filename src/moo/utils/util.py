import time

import numpy as np
import pandas as pd
from numpy.linalg import pinv
from tabulate import tabulate


def subroutine_times_problem(problem, results=None, repeat=10):
    times = []
    ts = []
    for i in range(repeat):
        t0 = time.process_time()
        problem.evaluate(problem.original_x)
        ts.append(round(time.process_time() - t0, 4))

    times.append(('f(x)', round(np.mean(ts), 4), round(np.std(ts), 4)))

    ts = []
    for i in range(repeat):
        t0 = time.process_time()
        dx = problem.gradient(problem.original_x)
        ts.append(round(time.process_time() - t0, 4))
    times.append(('J(x)', round(np.mean(ts), 4), round(np.std(ts), 4)))
    ts = []

    for i in range(repeat):
        t0 = time.process_time()
        dx_1 = pinv(dx)
        ts.append(round(time.process_time() - t0, 4))
    times.append(('J_1(x)', round(np.mean(ts), 4), round(np.std(ts), 4)))

    if results is not None:
        times.append(('execution', results['exec_time']))

    times = pd.DataFrame(times)
    times.set_index(0, inplace=True)
    times.columns = ['mean (s)', 'std (s)']
    times.index.name = 'routine'

    return times
