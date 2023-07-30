import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.problems.autodiff import AutomaticDifferentiation


class ContinuationProblem(ElementwiseProblem):

    def __init__(self,
                 f,
                 df,
                 n_var,
                 n_obj,
                 xl,
                 xu,
                 x_tol_for_hash=None,
                 constraints_limits=None):

        self.f = f
        self.df = df
        self.n_f_evals = 0
        self.n_grad_evals = 0
        self.computed_grads = {}
        self.x_tol_for_hash = x_tol_for_hash
        self.constraints_limits = constraints_limits
        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_constr=n_obj if constraints_limits is not None else 0,
                         xl=np.array(xl),
                         xu=np.array(xu))

    # wrapper to original method that returns type float32
    def evaluate32(self, X):
        res = super().evaluate(X)
        if isinstance(res, dict):
            for key, arr in res.items():
                res[key] = arr.astype(np.float32)
        else:
            res = res.astype(np.float32)
        return res

    def _evaluate(self, x, out, *args, **kwargs):
        self.n_f_evals += 1
        out["F"] = self.f(x)

        if self.constraints_limits is not None:
            F = out["F"] if len(out['F'].shape) == 2 else out["F"].reshape(1, -1)
            G = np.empty_like(F)
            if len(self.constraints_limits) != F.shape[1]:
                raise ValueError('{} constraints is not '
                                 'consistent with {} objectives'.format(len(self.constraints_limits),
                                                                        F.shape[1]))
            for obj in range(F.shape[1]):
                G[:, obj] = F[:, obj] - self.constraints_limits[obj]

            out['G'] = G

    def gradient(self, x):
        if self.x_tol_for_hash is None:
            self.n_grad_evals += 1
            return self.df(x)
        else:
            key = tuple(np.round(x, self.x_tol_for_hash))
            if key in self.computed_grads:
                return self.computed_grads[key]
            else:
                self.n_grad_evals += 1
                dx = self.df(x)
                self.computed_grads[key] = dx
                return dx


class AutomaticDifferentiationProblem(AutomaticDifferentiation):
    def __init__(self, problem):
        super().__init__(problem)

        self.n_f_evals = 0
        self.n_grad_evals = 0

    def gradient(self, x):
        return self.evaluate(x, return_values_of=['dF'])
