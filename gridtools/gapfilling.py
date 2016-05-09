import numpy as np
from numba import jit

import gridtools.resampling as gtr

_NOMASK = np.array(((False,),), dtype=np.bool)

DEFAULT_KERNEL = np.array([[0.5, 0.7, 0.5],
                           [0.7, 1.0, 0.7],
                           [0.5, 0.7, 0.5]])


def fillgaps_lowpass_2d(src, kernel=DEFAULT_KERNEL, threshold=1):
    w = src.shape[-1]
    h = src.shape[-2]
    pixel_count = w * h
    gap_count = 1
    out = src
    while 0 < gap_count < pixel_count:
        out, gap_count = _apply_low_pass_filter(out, kernel, threshold)
    return out


def fillgaps_multiscale_2d(src, ds_iter=True, ds_method=gtr.DS_MEAN, us_method=gtr.US_LINEAR):
    w = src.shape[-1]
    h = src.shape[-2]
    out = src
    pyramid = [src]
    s = 2
    while True:
        out_w = (w + s - 1) // s
        out_h = (h + s - 1) // s
        s *= 2
        out = gtr.downsample_2d(out if ds_iter else src, out_w, out_h, method=ds_method, fill_value=np.nan)
        pyramid.append(out)
        gap_count = count_gaps(out)
        if gap_count == 0 or gap_count == out_w * out_h or (out_w == 1 and out_h == 1):
            break
    pyramid.reverse()
    out_low = pyramid[0]
    for i in range(1, len(pyramid)):
        out_hi = pyramid[i]
        w = out_hi.shape[-1]
        h = out_hi.shape[-2]
        fill_data = gtr.upsample_2d(out_low, w, h, method=us_method, fill_value=np.nan)
        out_low, _ = _fill_gaps(out_hi, fill_data)
    return out_low


@jit(nopython=True)
def count_gaps(data):
    w = data.shape[-1]
    h = data.shape[-2]
    gap_count = 0
    for y in range(h):
        for x in range(w):
            if is_gap(data[y, x]):
                gap_count += 1
    return gap_count


@jit(nopython=True)
def is_gap(v):
    return not np.isfinite(v)


@jit(nopython=True)
def _apply_low_pass_filter(data, kernel, threshold):
    w = data.shape[-1]
    h = data.shape[-2]
    out = data.copy()
    kw = kernel.shape[-1]
    kh = kernel.shape[-2]
    kx0 = kw // 2
    ky0 = kh // 2
    gap_count = 0
    for y in range(h):
        for x in range(w):
            v = data[y, x]
            if is_gap(v):
                v_sum = 0.
                k_sum = 0.
                for ky in range(kh):
                    yy = y + ky - ky0
                    if 0 <= yy < h:
                        for kx in range(kw):
                            xx = x + kx - kx0
                            if 0 <= xx < w:
                                v = data[yy, xx]
                                if not is_gap(v):
                                    k = kernel[ky, kx]
                                    v_sum += k * v
                                    k_sum += k
                if k_sum != 0 and k_sum >= threshold:
                    out[y, x] = v_sum / k_sum
                else:
                    gap_count += 1

    return out, gap_count


@jit(nopython=True)
def _fill_gaps(data, fill_data):
    """
    Fills gap pixels by taking over values from a reduced resolution version of the grid.

    :param data: The data to gap-fill
    :param rr_data: the reduced resolution version.
    :return: a raster using the old raster data array instance, but gap-filled
    """
    out = data.copy()
    w = out.shape[-1]
    h = out.shape[-2]
    gap_count = 0
    for y in range(h):
        for x in range(w):
            if is_gap(out[y, x]):
                v = fill_data[y, x]
                if not is_gap(v):
                    out[y, x] = v
                else:
                    gap_count += 1
    return out, gap_count
