import numpy as np

from ._constants import *

try:
    from ._resample2d_numba import _upsample2d, _downsample2d, _resample2d
except ImportError:
    from ._resample2d_python import _upsample2d, _downsample2d, _resample2d


def _get_out(src, shape, out):
    if out is None:
        return np.zeros(shape, dtype=src.dtype)
    else:
        if out.shape != shape:
            raise ValueError("'shape' and 'out' are incompatible")
        if out.shape == src.shape:
            return None
        return out


def upsample2d(src, w, h, us_method=US_LINEAR, out=None):
    out = _get_out(src, (h, w), out)
    if out is None:
        return src
    return _upsample2d(src, us_method, out)


def downsample2d(src, w, h, ds_method=DS_MEAN, out=None):
    out = _get_out(src, (h, w), out)
    if out is None:
        return src
    return _downsample2d(src, ds_method, out)


def resample2d(src, w, h, ds_method=DS_MEAN, us_method=US_LINEAR, out=None):
    out = _get_out(src, (h, w), out)
    if out is None:
        return src
    return _resample2d(src, ds_method, us_method, out)


