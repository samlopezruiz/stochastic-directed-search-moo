# =========================================================================================================
# Imports
# =========================================================================================================
from pymoo.factory import get_from_list

from src.sds.core.corrector import DeltaCorrector, RankCorrector, ProjectionCorrector
from src.sds.core.predictor import StepAdjust, NoAdjustmentPredictors, LimitsPredictors
from src.sds.core.stepsize import Dominance, AngleBisection, Armijo, WeightedDominance, Angle
from src.sds.core.termination import MaxIter, Tol, NullTermination


# =========================================================================================================
# T Functions
# =========================================================================================================

def get_tfun_options():
    SAMPLING = [
        ("angle", Angle),
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
        ("projection", ProjectionCorrector),
        ("rank", RankCorrector),
        ("delta", DeltaCorrector)
    ]

    return SAMPLING


def get_corrector(name, *args, d={}, **kwargs):
    return get_from_list(get_corrector_options(), name, args, {**d, **kwargs})


# =========================================================================================================
# Termination
# =========================================================================================================

def get_termination_options():
    SAMPLING = [
        ("none", NullTermination),
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
        ("limit", LimitsPredictors),
    ]

    return SAMPLING


def get_predictor(name, *args, d={}, **kwargs):
    return get_from_list(get_predictor_options(), name, args, {**d, **kwargs})
