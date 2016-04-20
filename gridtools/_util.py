import warnings

NUMBA_DISABLED = False
# NUMBA_DISABLED = True

def jitify(*functions):
    if not NUMBA_DISABLED:
        try:
            import numba
            return [numba.jit(f, nopython=True) for f in functions]
        except ImportError:
            warnings.warn('numba not installed, using pure Python implementation')
    else:
        warnings.warn('numba disabled, using pure Python implementation')
    return functions
