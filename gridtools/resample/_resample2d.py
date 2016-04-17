from ._constants import *

try:
    from ._resample2d_numba import _upsample2d, _downsample2d
except ImportError:
    from ._resample2d_python import _upsample2d, _downsample2d


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


def _resample2d(src, ds_method, us_method, out):
    src_w = src.shape[-1]
    src_h = src.shape[-2]
    out_w = out.shape[-1]
    out_h = out.shape[-2]

    if out_w < src_w and out_h < src_h:
        return _downsample2d(src, ds_method, out)
    elif out_w < src_w:
        if out_h > src_h:
            temp = np.zeros((src_h, out_w), dtype=src.dtype)
            temp = _downsample2d(src, ds_method, temp)
            return _upsample2d(temp, us_method, out)
        else:
            return _downsample2d(src, ds_method, out)
    elif out_h < src_h:
        if out_w > src_w:
            temp = np.zeros((out_h, src_w), dtype=src.dtype)
            temp = _downsample2d(src, ds_method, temp)
            return _upsample2d(temp, us_method, out)
        else:
            return _downsample2d(src, ds_method, out)
    elif out_w > src_w or out_h > src_h:
        return _upsample2d(src, us_method, out)
    return src


def _get_out(src, shape, out):
    if out is None:
        return np.zeros(shape, dtype=src.dtype)
    else:
        if out.shape != shape:
            raise ValueError("'shape' and 'out' are incompatible")
        if out.shape == src.shape:
            return None
        return out
