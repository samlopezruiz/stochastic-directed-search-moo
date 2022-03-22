# =========================================================================================================
# Imports
# =========================================================================================================
from pymoo.factory import get_from_list

from src.moo.core.corrector import DsCorrector, DeltaCriteriaCorrector, DeltaCriteriaCorrectorValid, DsCorrector2
from src.moo.core.predictor import StepAdjust, NoAdjustmentPredictors
from src.moo.core.stepsize import Dominance, AngleBisection, Armijo, WeightedDominance
from src.moo.core.termination import MaxIter, Tol


# =========================================================================================================
# T Functions
# =========================================================================================================

def get_tfun_options():
    SAMPLING = [
        ("dominance", Dominance),
        ("angle_bisection", AngleBisection),
        ("armijo", Armijo),
        ("weighted_dominance", WeightedDominance),
    ]

    return SAMPLING


def get_tfun(name, *args, d={}, **kwargs):
    return get_from_list(get_tfun_options(), name, args, {**d, **kwargs})


# =========================================================================================================
# Correctors
# =========================================================================================================

def get_corrector_options():
    SAMPLING = [
        ("ds", DsCorrector2),
        ("delta_criteria", DeltaCriteriaCorrector),
        ("delta_valid", DeltaCriteriaCorrectorValid)
    ]

    return SAMPLING


def get_corrector(name, *args, d={}, **kwargs):
    return get_from_list(get_corrector_options(), name, args, {**d, **kwargs})


# =========================================================================================================
# Termination
# =========================================================================================================

def get_termination_options():
    SAMPLING = [
        ("n_iter", MaxIter),
        ("tol", Tol)
    ]

    return SAMPLING


def get_cont_termination(name, *args, d={}, **kwargs):
    return get_from_list(get_termination_options(), name, args, {**d, **kwargs})


# =========================================================================================================
# Correctors
# =========================================================================================================

def get_predictor_options():
    SAMPLING = [
        ("step_adjust", StepAdjust),
        ("no_adjustment", NoAdjustmentPredictors),
    ]

    return SAMPLING


def get_predictor(name, *args, d={}, **kwargs):
    return get_from_list(get_predictor_options(), name, args, {**d, **kwargs})
