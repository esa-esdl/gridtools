# http://stackoverflow.com/questions/7075082/what-is-future-in-python-used-for-and-how-when-to-use-it-and-how-it-works
from __future__ import division

import numpy as np

from ._constants import *


def _upsample2d(src, method, out):
    src_w = src.shape[-1]
    src_h = src.shape[-2]
    out_w = out.shape[-1]
    out_h = out.shape[-2]

    if src_w == out_w and src_h == out_h:
        return src

    if out_w < src_w or out_h < src_h:
        raise ValueError("Invalid target size")

    scale_x = (src_w - 1.0) / ((out_w - 1.0) if out_w > 1 else 1.0)
    scale_y = (src_h - 1.0) / ((out_h - 1.0) if out_h > 1 else 1.0)

    if method == US_NEAREST:
        for out_y in range(out_h):
            src_y = int(scale_y * out_y)
            for out_x in range(out_w):
                src_x = int(scale_x * out_x)
                out[out_y, out_x] = src[src_y, src_x]
    elif method == US_LINEAR:
        for out_y in range(out_h):
            src_yf = scale_y * out_y
            src_y = int(src_yf)
            wy = src_yf - src_y
            within_src_h = src_y + 1 < src_h
            for out_x in range(out_w):
                src_xf = scale_x * out_x
                src_x = int(src_xf)
                wx = src_xf - src_x
                within_src_w = src_x + 1 < src_w
                v00 = src[src_y, src_x]
                v01 = src[src_y, src_x + 1] if within_src_w else v00
                v10 = src[src_y + 1, src_x] if within_src_h else v00
                v11 = src[src_y + 1, src_x + 1] if within_src_w and within_src_h else v00
                v0 = v00 + wx * (v01 - v00)
                v1 = v10 + wx * (v11 - v10)
                v = v0 + wy * (v1 - v0)
                out[out_y, out_x] = v
    else:
        raise ValueError('invalid upsampling method: ' + str(method))
    return out


def _downsample2d(src, method, out):
    src_w = src.shape[-1]
    src_h = src.shape[-2]
    out_w = out.shape[-1]
    out_h = out.shape[-2]

    if src_w == out_w and src_h == out_h:
        return src

    if out_w > src_w or out_h > src_h:
        raise ValueError("Invalid target size")

    scale_x = src_w / out_w
    scale_y = src_h / out_h

    for out_y in range(out_h):
        src_yf0 = scale_y * out_y
        src_yf1 = src_yf0 + scale_y
        src_y0 = int(src_yf0)
        src_y1 = int(src_yf1)
        wy0 = 1.0 - (src_yf0 - src_y0)
        wy1 = src_yf1 - src_y1
        if wy1 < EPS:
            wy1 = 1.0
            if src_y1 > src_y0:
                src_y1 -= 1
        for out_x in range(out_w):
            src_xf0 = scale_x * out_x
            src_xf1 = src_xf0 + scale_x
            src_x0 = int(src_xf0)
            src_x1 = int(src_xf1)
            wx0 = 1.0 - (src_xf0 - src_x0)
            wx1 = src_xf1 - src_x1
            if wx1 < EPS:
                wx1 = 1.0
                if src_x1 > src_x0:
                    src_x1 -= 1
            v_sum = 0.0
            w_sum = 0.0
            for src_y in range(src_y0, src_y1 + 1):
                wy = wy0 if (src_y == src_y0) else wy1 if (src_y == src_y1) else 1.0
                for src_x in range(src_x0, src_x1 + 1):
                    wx = wx0 if (src_x == src_x0) else wx1 if (src_x == src_x1) else 1.0
                    v = src[src_y, src_x]
                    if not np.isnan(v):
                        w = wx * wy
                        v_sum += w * v
                        w_sum += w
            if np.isnan(v_sum) or w_sum < EPS:
                out[out_y, out_x] = np.nan
            else:
                out[out_y, out_x] = v_sum / w_sum
    return out


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
