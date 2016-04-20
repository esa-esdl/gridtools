import numpy as np

from .resampling import downsample2d, DS_MEAN


class Raster:
    def __init__(self, w, h, data, gap_count=None):
        self.w = w
        self.h = h
        self.data = data
        if gap_count is None:
            gap_count = 0
            for y in range(h):
                for x in range(w):
                    if Raster.is_gap(data[y, x]):
                        gap_count += 1
        self.gap_count = gap_count

    def is_full_of_gaps(self):
        return self.gap_count == self.w * self.h

    def is_free_of_gaps(self):
        return self.gap_count == 0

    def is_singular(self):
        return self.w == 1 and self.h == 1

    @staticmethod
    def is_gap(v):
        return not np.isfinite(v)


GF_AVG_NEAREST_CONTINUED = 10
GF_AVG_NEAREST_DOWNSAMPLE = 20


def fillgaps2d(data, method=GF_AVG_NEAREST_DOWNSAMPLE, valid_count_min=1):
    w = data.shape[-1]
    h = data.shape[-2]

    if w == 1 and h == 1:
        return data

    if method == GF_AVG_NEAREST_CONTINUED:
        return fill_gaps_1(w, h, data, valid_count_min)
    elif method == GF_AVG_NEAREST_DOWNSAMPLE:
        return fill_gaps_2(w, h, data, valid_count_min)
    else:
        raise ValueError('invalid method')


def fill_gaps_1(w, h, data, valid_count_min):
    if w == 1 and h == 1:
        return data

    raster = Raster(w, h, data)
    if raster.is_full_of_gaps():
        return data

    while not raster.is_free_of_gaps():
        raster = _fill_gaps_by_surroundings(raster, valid_count_min)

    return raster.data


def fill_gaps_2(w, h, data, valid_count_min):
    if w == 1 and h == 1:
        return data

    raster = _fill_gaps_2(Raster(w, h, data), valid_count_min)
    return raster.data


def _fill_gaps_2(raster, valid_count_min):
    if raster.is_singular() or raster.is_full_of_gaps() or raster.is_free_of_gaps():
        return raster

    # Note, raster is a copy
    raster = _fill_gaps_by_surroundings(raster, valid_count_min)
    if raster.is_free_of_gaps():
        return raster

    downsampled = downsample2d(raster.data,
                               (raster.w + 1) / 2,
                               (raster.h + 1) / 2,
                               method=DS_MEAN,
                               fill_value=np.nan)
    downsampled = _fill_gaps_2(Raster(downsampled.shape[-1], downsampled.shape[-2], downsampled), valid_count_min)
    raster = _fill_gaps_from_downsampled(raster, downsampled)

    return raster


def _fill_gaps_from_downsampled(raster, downsampled_raster):
    """
    Fills gap pixels by taking over values from a downsampled version of the raster.

    :param raster: The raster to gap-fill
    :param downsampled_raster: the downsampled version.
    :return: a raster using  the old raster data array instance, but gap-filled
    """
    out_w = raster.w
    out_h = raster.h
    out = raster.data
    src = downsampled_raster.data
    filled_count = 0
    for out_y in range(out_h):
        for out_x in range(out_w):
            if Raster.is_gap(out[out_y, out_x]):
                src_x = out_x / 2
                src_y = out_y / 2
                v = src[src_y, src_x]
                if not Raster.is_gap(v):
                    out[out_y, out_x] = v
                    filled_count += 1

    return Raster(out_w, out_h, np.array(out), raster.gap_count - filled_count)


def _fill_gaps_by_surroundings(raster, valid_count_min):
    """
    /**
     * Fills gap pixels by averaging the surrounding pixels, if any.
     *
     * @param raster source raster
     * @return gap-filled raster, always uses a array instance
     */
    :return:
    """
    w = raster.w
    h = raster.h
    data = raster.data
    gap_filled_data = np.array(data)
    filled_count = 0
    for y in range(h):
        for x in range(w):
            if Raster.is_gap(data[y, x]):
                x1 = x - 1 if x > 0 else x
                y1 = y - 1 if y > 0 else y
                x2 = x + 1 if x < w - 1 else x
                y2 = y + 1 if y < h - 1 else y
                v_sum = 0
                valid_count = 0
                for yy in range(y1, y2 + 1):
                    for xx in range(x1, x2 + 1):
                        v = data[yy, xx]
                        if not Raster.is_gap(v):
                            v_sum += v
                            valid_count += 1

                if valid_count >= valid_count_min:
                    gap_filled_data[y, x] = v_sum / valid_count
                    filled_count += 1

    return Raster(w, h, gap_filled_data, raster.gap_count - filled_count)
