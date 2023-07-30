import numpy as np
from pymoo.factory import get_performance_indicator


def hypervolume(F, ref=None):
    ref = np.max(F, axis=0) if ref is None else np.array(ref)
    hv = get_performance_indicator("hv", ref_point=ref)
    return hv.do(F)