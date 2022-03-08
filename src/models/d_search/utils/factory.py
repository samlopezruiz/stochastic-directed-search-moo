
# =========================================================================================================
# Generic
# =========================================================================================================
from pymoo.factory import get_from_list

from src.models.d_search.algorithms.corrector import DsCorrector, DeltaCriteriaCorrector
from src.models.d_search.algorithms.predictor import StepAdjust, NoAdjustmentPredictors
from src.models.d_search.algorithms.stepsize import Dominance, AngleBisection, Armijo, WeightedDominance

# =========================================================================================================
# T Functions
# =========================================================================================================
from src.models.d_search.algorithms.termination import MaxIter, Tol


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
        ("ds", DsCorrector),
        ("delta_criteria", DeltaCriteriaCorrector)
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


