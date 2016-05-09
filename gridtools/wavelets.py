import numpy as np
import gridtools.gapfilling as gtg

def get_filter_pair(s):
    """
    :param s: Scaling function coefficients
    :return: Tuple of scaling and wavelet function coefficients
    """
    s = np.array(s, dtype=np.float64)
    s /= np.sqrt((s * s).sum())
    w = np.array([((-1) ** i) * s[-1 - i] for i in range(len(s))])
    return s, w


D2 = get_filter_pair([1, 1])
D4 = get_filter_pair([0.6830127, 1.1830127, 0.3169873, -0.1830127])


def transform_2d(decomp, levels, t):
    indexes = get_indices(decomp.shape, levels)
    result = np.zeros(decomp.shape, dtype=decomp.dtype)
    result[indexes[0]] = t.low_low(decomp[indexes[0]])
    i = 1
    for level in range(levels):
        result[indexes[i + 0]] = t.hi_low(decomp[indexes[i + 0]])
        result[indexes[i + 1]] = t.low_hi(decomp[indexes[i + 1]])
        result[indexes[i + 2]] = t.hi_hi(decomp[indexes[i + 2]])
        i += 3
    return result


def get_indices(shape, levels):
    if levels < 1:
        raise ValueError('invalid level')
    dim = len(shape)
    if dim < 1 or dim > 2:
        raise ValueError('invalid shape')
    indices = []
    if len(shape) == 1:
        width = shape[-1]
        for level in range(levels):
            width //= 2
            indices.append(slice(width, 2 * width))
            if level == levels - 1:
                indices.append(slice(0, width))
    elif len(shape) == 2:
        width = shape[-1]
        height = shape[-2]
        indices = []
        for level in range(levels):
            width //= 2
            height //= 2
            indices.append((slice(height, 2 * height), slice(width, 2 * width)))
            indices.append((slice(height, 2 * height), slice(0, width)))
            indices.append((slice(0, height), slice(width, 2 * width)))
            if level == levels - 1:
                indices.append((slice(0, height), slice(0, width)))
    indices.reverse()
    return indices


def wt_analyse_2d(signal, filter=D2, max_level=1):
    coeff_lo, coeff_hi = filter
    decomp = np.zeros(signal.shape, dtype=signal.dtype)
    tmp = signal.copy()
    filter_size = len(coeff_lo)
    width = signal.shape[-1]
    height = signal.shape[-2]
    sub_width = width // 2
    sub_height = height // 2
    for level in range(max_level):
        if sub_width == 0 or sub_height == 0:
            break
        for row in range(2 * sub_height):
            for col in range(sub_width):
                sum_lo = 0
                sum_hi = 0
                for i in range(filter_size):
                    col2 = 2 * col + i
                    if col2 >= width:
                        col2 = width - 1
                    value = tmp[row, col2]
                    sum_lo += coeff_lo[i] * value
                    sum_hi += coeff_hi[i] * value
                decomp[row, col] = sum_lo
                decomp[row, sub_width + col] = sum_hi
            tmp[row, :] = decomp[row, :]
        for col in range(2 * sub_width):
            for row in range(sub_height):
                sum_lo = 0
                sum_hi = 0
                for i in range(filter_size):
                    row2 = 2 * row + i
                    if row2 >= height:
                        row2 = height - 1
                    value = tmp[row2, col]
                    sum_lo += coeff_lo[i] * value
                    sum_hi += coeff_hi[i] * value
                decomp[row, col] = sum_lo
                decomp[sub_height + row, col] = sum_hi
            tmp[:sub_height, col] = decomp[:sub_height, col]
        sub_width //= 2
        sub_height //= 2
    return decomp


def wt_synthesize_2d(decomp, filter=D2, max_level=1):
    coeff_lo, coeff_hi = filter
    signal = decomp.copy()
    tmp_lo = np.zeros(decomp.shape, dtype=decomp.dtype)
    tmp_hi = np.zeros(decomp.shape, dtype=decomp.dtype)
    filter_size = len(coeff_lo)
    width = signal.shape[-1]
    height = signal.shape[-2]
    sub_width = width // (2 ** max_level)
    sub_height = height // (2 ** max_level)
    for level in range(max_level):
        for col in range(2 * sub_width):
            for row in range(sub_height):
                value_lo = signal[row, col]
                value_hi = signal[sub_height + row, col]
                for i in range(filter_size):
                    row2 = 2 * row + i
                    # if row2 >= height:
                    #    row2 = height - 1
                    if row2 < height:
                        tmp_lo[row2, col] = coeff_lo[i] * value_lo
                        tmp_hi[row2, col] = coeff_hi[i] * value_hi
            for row in range(2 * sub_height):
                signal[row, col] = tmp_lo[row, col] + tmp_hi[row, col]
        for row in range(2 * sub_height):
            for col in range(sub_width):
                value_lo = signal[row, col]
                value_hi = signal[row, sub_width + col]
                for i in range(filter_size):
                    col2 = 2 * col + i
                    # if col2 >= width:
                    #    col2 = width - 1
                    if col2 < width:
                        tmp_lo[row, col2] = coeff_lo[i] * value_lo
                        tmp_hi[row, col2] = coeff_hi[i] * value_hi
            for col in range(2 * sub_width):
                signal[row, col] = tmp_lo[row, col] + tmp_hi[row, col]
        sub_width *= 2
        sub_height *= 2
    return signal


def wt_analyse_1d(signal, filter=D2, max_level=1):
    coeff_lo, coeff_hi = filter
    decomp = np.zeros(signal.shape, dtype=signal.dtype)
    tmp = signal.copy()
    filter_size = len(coeff_lo)
    width = signal.shape[-1]
    sub_width = width // 2
    for level in range(max_level):
        if sub_width == 0:
            break
        for col in range(sub_width):
            sum_lo = 0
            sum_hi = 0
            for i in range(filter_size):
                col2 = 2 * col + i
                if col2 >= width:
                    col2 = width - 1
                value = tmp[col2]
                sum_lo += coeff_lo[i] * value
                sum_hi += coeff_hi[i] * value
            # print(level, i, n + i, m * i)
            decomp[col] = sum_lo
            decomp[sub_width + col] = sum_hi
        tmp[:sub_width] = decomp[:sub_width]
        sub_width //= 2
    return decomp


def wt_synthesize_1d(decomp, filter=D2, max_level=1):
    coeff_lo, coeff_hi = filter
    signal = decomp.copy()
    tmp_lo = np.zeros(decomp.shape, dtype=decomp.dtype)
    tmp_hi = np.zeros(decomp.shape, dtype=decomp.dtype)
    filter_size = len(coeff_lo)
    width = signal.shape[-1]
    sub_width = width // (2 ** max_level)
    for level in range(max_level):
        for col in range(sub_width):
            value_lo = signal[col]
            value_hi = signal[sub_width + col]
            for i in range(filter_size):
                col2 = 2 * col + i
                if col2 >= width:
                    col2 = width - 1
                tmp_lo[col2] = coeff_lo[i] * value_lo
                tmp_hi[col2] = coeff_hi[i] * value_hi
        sub_width *= 2
        for col in range(sub_width):
            signal[col] = tmp_lo[col] + tmp_hi[col]
    return signal


if __name__ == "__main__":
    from PIL import Image


    def _save_image(a, path):
        image = Image.fromarray(np.uint8(np.array(a)))
        image.save(path)


    def _erase_circle(im, cx, cy, cr):
        for y in range(im.shape[-2]):
            for x in range(im.shape[-1]):
                dx = (x - cx)
                dy = (y - cy)
                if dx * dx + dy * dy < cr * cr:
                    im[y, x] = np.nan


    def _erase_rect(im, rx, ry, rw, rh):
        for y in range(im.shape[-2]):
            for x in range(im.shape[-1]):
                if rx <= x <= rx + rw and ry <= y <= ry + rh:
                    im[y, x] = np.nan


    im = Image.open("notebooks/houses-512.png")
    signal = np.float64(np.array(im.getdata(band=0)).reshape((im.height, im.width)))

    _erase_circle(signal, 100, 100, 30)
    _erase_rect(signal, 200, 300, 200, 50)

    _save_image(signal, "signal_512.png")

    nogaps, _ = gtg.fillgaps2d(signal, method=gtg.GF_PYRAMID2)
    _save_image(nogaps, "nogaps_512.png")

    FILTER = D2
    LEVELS = 6

    decomp = wt_analyse_2d(signal, filter=FILTER, max_level=LEVELS)


    class T:
        def low_low(self, a):
            # b, _ = gtg.fillgaps2d(a, method=gtg.GF_MEAN_OF_NEAREST_MULTI_SCALE)
            # return b
            return self.fillnan(a)

        def hi_low(self, a):
            return self.fillnan(a)

        def low_hi(self, a):
            return self.fillnan(a)

        def hi_hi(self, a):
            return self.fillnan(a)

        def fillnan(self, a):
            #b = a.copy()
            #b[np.where(np.isnan(a))] = np.nanmean(a)
            # return np.zeros(a.shape, dtype=a.dtype)
            b, _ = gtg.fillgaps2d(a, method=gtg.GF_PYRAMID2)
            return b


    decomp = transform_2d(decomp, LEVELS, T())

    # _save_image(255 * (decomp - decomp.min()) / (decomp.max() - decomp.min()), "decomp_512.png")
    _save_image(decomp, "decomp_512.png")

    signal2 = wt_synthesize_2d(decomp, filter=FILTER, max_level=LEVELS)

    _save_image(signal2, "signal2_512.png")
