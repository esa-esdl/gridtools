import numpy as np
from numba import jit

#from ._util import jitify
#from .resampling import downsample2d, DS_MEAN

import gridtools.resampling as gtr

GF_MEAN_OF_NEAREST = 10
GF_MEAN_OF_NEAREST_MULTI_SCALE = 20


def fillgaps2d(data, method=GF_MEAN_OF_NEAREST_MULTI_SCALE, valid_count_min=1):
    w = data.shape[-1]
    h = data.shape[-2]

    if w <= 1 and h <= 1:
        return data, 0

    gap_count = _count_gaps(data)
    if gap_count == 0 or gap_count == w * h:
        return data, 0

    if method == GF_MEAN_OF_NEAREST:
        data_filled, gaps_remaining = _fill_gaps_by_mean_of_nearest_iter(data, valid_count_min)
    elif method == GF_MEAN_OF_NEAREST_MULTI_SCALE:
        data_filled, gaps_remaining = _fill_gaps_by_mean_of_nearest_on_multi_scales(data, valid_count_min)
    else:
        raise ValueError('invalid gap-filling method')

    return data_filled, gap_count - gaps_remaining


@jit(nopython=True)
def _fill_gaps_by_mean_of_nearest_iter(data, valid_count_min):
    gap_count = 1
    while gap_count > 0:
        data, gap_count = _fill_gaps_by_mean_of_nearest(data, valid_count_min)
    return data, gap_count


@jit(nopython=True)
def _fill_gaps_by_mean_of_nearest_on_multi_scales(data, valid_count_min):
    w = data.shape[-1]
    h = data.shape[-2]

    data, gap_count = _fill_gaps_by_mean_of_nearest(data, valid_count_min)
    if gap_count == 0 or gap_count == data.size:
        return data, gap_count

    lowres_data = gtr.downsample2d(data,
                               (w + 1) / 2,
                               (h + 1) / 2,
                               method=gtr.DS_MEAN,
                               fill_value=np.nan)

    # Recursion until no more gaps or everything is a gap
    lowres_data, lowres_gap_count = _fill_gaps_by_mean_of_nearest_on_multi_scales(lowres_data, valid_count_min)
    if lowres_gap_count == lowres_data.size:
        # If the low-res version entirely consists of gaps, we cannot get any better
        return data, lowres_gap_count
    else:
        # Otherwise fill gaps from new valid low resolution cells
        return _fill_gaps_from_valid_lowres_cells(data, lowres_data)


@jit(nopython=True)
def _fill_gaps_from_valid_lowres_cells(data, lowres_data):
    """
    Fills gap pixels by taking over values from a downsampled version of the raster.

    :param data: The data to gap-fill
    :param lowres_data: the downsampled version.
    :return: a raster using the old raster data array instance, but gap-filled
    """
    out = np.array(data)
    out_w = out.shape[-1]
    out_h = out.shape[-2]
    gap_count = 0
    for out_y in range(out_h):
        for out_x in range(out_w):
            if _is_gap(out[out_y, out_x]):
                src_x = out_x / 2
                src_y = out_y / 2
                v = lowres_data[src_y, src_x]
                if not _is_gap(v):
                    out[out_y, out_x] = v
                else:
                    gap_count += 1

    return out, gap_count


@jit(nopython=True)
def _fill_gaps_by_mean_of_nearest(src, valid_count_min):
    """
    /**
     * Fills gap pixels by averaging the surrounding pixels, if any.
     *
     * @param raster source raster
     * @return gap-filled raster, always uses a array instance
     */
    :return:
    """
    w = src.shape[-1]
    h = src.shape[-2]
    out = np.array(src)
    gap_count = 0
    for y in range(h):
        for x in range(w):
            if _is_gap(src[y, x]):
                x1 = x - 1 if x > 0 else x
                y1 = y - 1 if y > 0 else y
                x2 = x + 1 if x < w - 1 else x
                y2 = y + 1 if y < h - 1 else y
                v_sum = 0
                valid_count = 0
                for yy in range(y1, y2 + 1):
                    for xx in range(x1, x2 + 1):
                        v = src[yy, xx]
                        if not _is_gap(v):
                            v_sum += v
                            valid_count += 1
                if valid_count >= valid_count_min:
                    out[y, x] = v_sum / valid_count
                else:
                    gap_count += 1
    return out, gap_count


@jit(nopython=True)
def _count_gaps(raster):
    w = raster.shape[-1]
    h = raster.shape[-2]
    gap_count = 0
    for y in range(h):
        for x in range(w):
            if _is_gap(raster[y, x]):
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
