# from http://www.olhovsky.com/content/wavelet/2dwavelet97lift.py


'''
2D CDF 9/7 Wavelet Forward and Inverse Transform (lifting implementation)

This code is provided "as is" and is given for educational purposes.
2008 - Kris Olhovsky - code.inquiries@olhovsky.com
'''

from PIL import Image  # Part of the standard Python Library

# 9/7 Coefficients:
WT97_A1 = -1.586134342
WT97_A2 = -0.05298011854
WT97_A3 = 0.8829110762
WT97_A4 = 0.4435068522

# Scale coeff:
WT97_K1 = 0.81289306611596146  # 1/1.230174104914
WT97_K2 = 0.61508705245700002  # 1.230174104914/2


def fwt97_2d(m, nlevels=1):
    """
    Perform the CDF 9/7 transform on a 2D matrix signal m.
    nlevel is the desired number of times to recursively transform the
    signal.
    """

    w = m.shape[-1]
    h = m.shape[-2]
    for i in range(nlevels):
        m = fwt97(m, w, h)  # cols
        m = fwt97(m, w, h)  # rows
        w //= 2
        h //= 2

    return m


def iwt97_2d(m, nlevels=1):
    """
    Inverse CDF 9/7 transform on a 2D matrix signal m.
    nlevels must be the same as the nlevels used to perform the fwt.
    """

    w = m.shape[-1]
    h = m.shape[-2]

    # Find starting size of m:
    for i in range(nlevels - 1):
        h //= 2
        w //= 2

    for i in range(nlevels):
        m = iwt97(m, w, h)  # rows
        m = iwt97(m, w, h)  # cols
        h *= 2
        w *= 2

    return m


def fwt97(s, width, height):
    """
    Forward Cohen-Daubechies-Feauveau 9 tap / 7 tap wavelet transform
    performed on all columns of the 2D n*n matrix signal s via lifting.
    The returned result is s, the modified input matrix.
    The highpass and lowpass results are stored on the left half and right
    half of s respectively, after the matrix is transposed.
    """

    for col in range(width):  # Do the 1D transform on all cols:
        ''' Core 1D lifting process in this loop. '''
        ''' Lifting is done on the cols. '''

        # Predict 1. y1
        for row in range(1, height - 1, 2):
            v1 = s[row - 1, col]
            v2 = s[row + 1, col]
            vsum = v1 + v2
            if np.isnan(vsum):
                if np.isnan(v1) and np.isnan(v2):
                    vsum = 0
                elif np.isnan(v1):
                    vsum = 2 * v2
                else:
                    vsum = 2 * v1
            s[row, col] += WT97_A1 * vsum
        vsum = 2 * s[height - 2, col]
        if np.isnan(vsum):
            vsum = 0
        s[height - 1, col] += WT97_A1 * vsum  # Symmetric extension

        # Update 1. y0
        for row in range(2, height, 2):
            v1 = s[row - 1, col]
            v2 = s[row + 1, col]
            vsum = v1 + v2
            if np.isnan(vsum):
                if np.isnan(v1) and np.isnan(v2):
                    vsum = 0
                elif np.isnan(v1):
                    vsum = 2 * v2
                else:
                    vsum = 2 * v1
            s[row, col] += WT97_A2 * vsum
        vsum = 2 * s[1, col]
        if np.isnan(vsum):
            vsum = 0
        s[0, col] += WT97_A2 * vsum  # Symmetric extension

        # Predict 2.
        for row in range(1, height - 1, 2):
            v1 = s[row - 1, col]
            v2 = s[row + 1, col]
            vsum = v1 + v2
            if np.isnan(vsum):
                if np.isnan(v1) and np.isnan(v2):
                    vsum = 0
                elif np.isnan(v1):
                    vsum = 2 * v2
                else:
                    vsum = 2 * v1
            s[row, col] += WT97_A3 * vsum
        vsum = 2 * s[height - 2, col]
        if np.isnan(vsum):
            vsum = 0
        s[height - 1, col] += WT97_A3 * vsum

        # Update 2.
        for row in range(2, height, 2):
            v1 = s[row - 1, col]
            v2 = s[row + 1, col]
            vsum = v1 + v2
            if np.isnan(vsum):
                if np.isnan(v1) and np.isnan(v2):
                    vsum = 0
                elif np.isnan(v1):
                    vsum = 2 * v2
                else:
                    vsum = 2 * v1
            s[row, col] += WT97_A4 * vsum
        vsum = 2 * s[1, col]
        if np.isnan(vsum):
            vsum = 0
        s[0, col] += WT97_A4 * vsum

    # de-interleave
    temp_bank = np.zeros((height, width), dtype=np.float64)
    for row in range(height):
        for col in range(width):
            # WT97_K1 and k2 scale the vals
            # simultaneously transpose the matrix when deinterleaving
            if row % 2 == 0:  # even
                temp_bank[col, row // 2] = WT97_K1 * s[row, col]
            else:  # odd
                temp_bank[col, row // 2 + height // 2] = WT97_K2 * s[row, col]

    # write temp_bank to s:
    for row in range(width):
        for col in range(height):
            s[row, col] = temp_bank[row, col]

    return s


def iwt97(s, width, height):
    """
    Inverse CDF 9/7.
    """

    # 9/7 inverse coefficients:
    a1_inv = -WT97_A1
    a2_inv = -WT97_A2
    a3_inv = -WT97_A3
    a4_inv = -WT97_A4

    # Inverse scale coeffs:
    k1_inv = 1. / WT97_K1
    k2_inv = 1. / WT97_K2

    # Interleave:
    temp_bank = np.zeros((height, width), dtype=np.float64)
    for col in range(width // 2):
        for row in range(height):
            # k1 and k2 scale the vals
            # simultaneously transpose the matrix when interleaving
            temp_bank[col * 2, row] = k1_inv * s[row, col]
            temp_bank[col * 2 + 1, row] = k2_inv * s[row, col + width // 2]

    # write temp_bank to s:
    for row in range(width):
        for col in range(height):
            s[row, col] = temp_bank[row, col]

    for col in range(width):  # Do the 1D transform on all cols:
        ''' Perform the inverse 1D transform. '''

        # Inverse update 2.
        for row in range(2, height, 2):
            s[row, col] += a4_inv * (s[row - 1, col] + s[row + 1, col])
        s[0, col] += 2 * a4_inv * s[1, col]

        # Inverse predict 2.
        for row in range(1, height - 1, 2):
            s[row, col] += a3_inv * (s[row - 1, col] + s[row + 1, col])
        s[height - 1, col] += 2 * a3_inv * s[height - 2, col]

        # Inverse update 1.
        for row in range(2, height, 2):
            s[row, col] += a2_inv * (s[row - 1, col] + s[row + 1, col])
        s[0, col] += 2 * a2_inv * s[1, col]  # Symmetric extension

        # Inverse predict 1.
        for row in range(1, height - 1, 2):
            s[row, col] += a1_inv * (s[row - 1, col] + s[row + 1, col])
        s[height - 1, col] += 2 * a1_inv * s[height - 2, col]  # Symmetric extension

    return s


def array_to_image(m):
    return Image.fromarray(np.uint8(np.array(m)))


import numpy as np

if __name__ == "__main__":
    # Load image.
    im = Image.open("notebooks/houses-512.png")  # Must be a single band image! (grey)

    # Create an image buffer object for fast access.
    pix = im.load()

    # Convert the 2d image to a 1d sequence:
    m = np.float64(np.array(im.getdata(band=0)).reshape((im.height, im.width)))

    LEVELS = 5

    # Save the transformed image.
    im = array_to_image(255 * (m - m.min()) / (m.max() - m.min()))
    im.save("test1_512_fwt.png")

    # Perform an inverse transform:
    m = iwt97_2d(m, LEVELS)

    # Save the inverse transformation.
    im = array_to_image(m)
    im.save("test1_512_iwt.png")
