import numpy as np
from numba import jit

import gridtools.resampling as gtr

GF_MEAN_OF_NEAREST = 10
GF_MEAN_OF_NEAREST_MULTI_SCALE = 20

_NOMASK = np.array(((False,),), dtype=np.bool)


def fillgaps2d(data, method=GF_MEAN_OF_NEAREST_MULTI_SCALE, valid_count_min=1):
    w = data.shape[-1]
    h = data.shape[-2]

    if w <= 1 and h <= 1:
        return data, 0

    gap_count_initial = _count_gaps(data)
    if gap_count_initial == 0 or gap_count_initial == w * h:
        return data, 0

    if method == GF_MEAN_OF_NEAREST:
        data_filled, gap_count = _fill_gaps_by_mean_of_nearest_iter(data, valid_count_min)
    elif method == GF_MEAN_OF_NEAREST_MULTI_SCALE:
        data_filled, gap_count = _fill_gaps_by_mean_of_nearest_on_multi_scales(data, valid_count_min)
    else:
        raise ValueError('invalid gap-filling method')

    return data_filled, gap_count_initial - gap_count


@jit(nopython=True)
def _fill_gaps_by_mean_of_nearest_iter(data, valid_count_min):
    gap_count = 1
    while gap_count > 0:
        data, gap_count = _fill_gaps_by_mean_of_nearest(data, valid_count_min)
    return data, gap_count


# Note: jit has been turned off here, because of https://github.com/numba/numba/issues/1845 in Numba 0.24:
# "Compilation of recursive function fails with nopython=True"
#
# @jit(nopython=True)
def _fill_gaps_by_mean_of_nearest_on_multi_scales(data, valid_count_min):
    w = data.shape[-1]
    h = data.shape[-2]

    data, gap_count = _fill_gaps_by_mean_of_nearest(data, valid_count_min)
    if gap_count == 0 or gap_count == w * h:
        return data, gap_count

    # Compute reduced resolution grid
    # rr_data, _ = _reduce_res(data)
    rr_w = (w + 1) // 2
    rr_h = (h + 1) // 2
    rr_data = np.zeros((rr_h, rr_w), dtype=data.dtype)
    rr_data = gtr._downsample2d(data,
                                _NOMASK,
                                False,
                                gtr.DS_MEAN,
                                np.nan,
                                rr_data)

    # Recursion until no more gaps or only gaps
    rr_data, rr_gap_count = _fill_gaps_by_mean_of_nearest_on_multi_scales(rr_data, valid_count_min)
    if rr_gap_count == rr_w * rr_h:
        # If the red-res version entirely consists of gaps, we cannot get any better
        return data, gap_count
    else:
        # Otherwise fill gaps from new valid low resolution cells
        return _fill_gaps_from_valid_lowres_cells(data, rr_data)


@jit(nopython=True)
def _fill_gaps_from_valid_lowres_cells(data, rr_data):
    """
    Fills gap pixels by taking over values from a reduced resolution version of the grid.

    :param data: The data to gap-fill
    :param rr_data: the reduced resolution version.
    :return: a raster using the old raster data array instance, but gap-filled
    """
    out = data.copy()
    out_w = out.shape[-1]
    out_h = out.shape[-2]
    gap_count = 0
    for out_y in range(out_h):
        for out_x in range(out_w):
            if _is_gap(out[out_y, out_x]):
                src_x = out_x // 2
                src_y = out_y // 2
                v = rr_data[src_y, src_x]
                if not _is_gap(v):
                    out[out_y, out_x] = v
                else:
                    gap_count += 1
    return out, gap_count


@jit(nopython=True)
def _fill_gaps_by_mean_of_nearest(data, valid_count_min):
    w = data.shape[-1]
    h = data.shape[-2]
    out = data.copy()
    gap_count = 0
    for y in range(h):
        for x in range(w):
            if _is_gap(data[y, x]):
                x1 = x - 1 if x > 0 else x
                y1 = y - 1 if y > 0 else y
                x2 = x + 1 if x < w - 1 else x
                y2 = y + 1 if y < h - 1 else y
                v_sum = 0
                valid_count = 0
                for yy in range(y1, y2 + 1):
                    for xx in range(x1, x2 + 1):
                        v = data[yy, xx]
                        if not _is_gap(v):
                            v_sum += v
                            valid_count += 1
                if valid_count >= valid_count_min:
                    out[y, x] = v_sum / valid_count
                else:
                    gap_count += 1
    return out, gap_count


# Not used here anymore, but I will add this to resampling.py for optimized downsampling if w % 2 == 0 and h % 2 == 0
@jit(nopython=True)
def _reduce_res(src):
    src_w = src.shape[-1]
    src_h = src.shape[-2]
    out_w = (src_w + 1) // 2
    out_h = (src_h + 1) // 2
    out = np.zeros((out_h, out_w), dtype=src.dtype)
    gap_count = 0
    for out_y in range(out_h):
        src_y = out_y // 2
        within_h = src_y < src_h - 1
        for out_x in range(out_w):
            src_x = out_x // 2
            within_w = src_x < src_w - 1
            v_sum = 0.0
            v_num = 0
            v = src[src_y, src_x]
            if not _is_gap(v):
                v_sum += v
                v_num += 1
            if within_w:
                v = src[src_y, src_x + 1]
                if not _is_gap(v):
                    v_sum += v
                    v_num += 1
                if within_h:
                    v = src[src_y + 1, src_x + 1]
                    if not _is_gap(v):
                        v_sum += v
                        v_num += 1
            if within_h:
                v = src[src_y + 1, src_x]
                if not _is_gap(v):
                    v_sum += v
                    v_num += 1
            if v_num != 0:
                out[out_y, out_x] = v_sum / v_num
            else:
                out[out_y, out_x] = np.nan
                gap_count += 1
    return out, gap_count


@jit(nopython=True)
def _count_gaps(data):
    w = data.shape[-1]
    h = data.shape[-2]
    gap_count = 0
    for y in range(h):
        for x in range(w):
            if _is_gap(data[y, x]):
                gap_count += 1
    return gap_count


@jit(nopython=True)
def _is_gap(v):
    return not np.isfinite(v)

#
# _fill_gaps_by_mean_of_nearest_iter, \
# _fill_gaps_by_mean_of_nearest_on_multi_scales, \
# _fill_gaps_from_valid_lowres_cells, \
# _fill_gaps_by_mean_of_nearest, \
# _count_gaps, \
# _is_gap = \
#     jitify(_fill_gaps_by_mean_of_nearest_iter,
#            _fill_gaps_by_mean_of_nearest_on_multi_scales,
#            _fill_gaps_from_valid_lowres_cells,
#            _fill_gaps_by_mean_of_nearest,
#            _count_gaps,
#            _is_gap)
