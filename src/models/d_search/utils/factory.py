
# =========================================================================================================
# Generic
# =========================================================================================================
from pymoo.factory import get_from_list

from src.models.d_search.algorithms.corrector import DsCorrector
from src.models.d_search.algorithms.predictor import LeftPredictor
from src.models.d_search.algorithms.stepsize import Szc5


# =========================================================================================================
# T Functions
# =========================================================================================================

def get_tfun_options():

    SAMPLING = [
        ("szc5", Szc5),
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
    ]

    return SAMPLING


def get_corrector(name, *args, d={}, **kwargs):
    return get_from_list(get_corrector_options(), name, args, {**d, **kwargs})



# =========================================================================================================
# Correctors
# =========================================================================================================

def get_predictor_options():

    SAMPLING = [
        ("left", LeftPredictor),
    ]

    return SAMPLING


def get_predictor(name, *args, d={}, **kwargs):
    return get_from_list(get_predictor_options(), name, args, {**d, **kwargs})


