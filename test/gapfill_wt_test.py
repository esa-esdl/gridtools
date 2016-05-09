import numpy as np
from PIL import Image

import gridtools.wavelet97lift2d_nan as wt


def array_to_image(m):
    return Image.fromarray(np.uint8(np.array(m)))


def _erase_circle(im, cx, cy, cr):
    for y in range(m.shape[-2]):
        for x in range(m.shape[-1]):
            dx = (x - cx)
            dy = (y - cy)
            if dx * dx + dy * dy < cr * cr:
                im[y, x] = np.nan


def _erase_rect(im, rx, ry, rw, rh):
    for y in range(m.shape[-2]):
        for x in range(m.shape[-1]):
            if rx <= x <= rx + rw and ry <= y <= ry + rh:
                im[y, x] = np.nan


if __name__ == "__main__":
    # Load image.
    im = Image.open("notebooks/houses-512.png")  # Must be a single band image! (grey)

    # Create an image buffer object for fast access.
    pix = im.load()

    # Convert the 2d image to a 1d sequence:
    m = np.float64(np.array(im.getdata(band=0)).reshape((im.height, im.width)))

    _erase_circle(m, 100, 100, 30)
    _erase_rect(m, 200, 300, 200, 50)

    im = array_to_image(m)
    im.save("test1_512_holes.png")

    LEVELS = 4

    # Perform a forward CDF 9/7 transform on the image:
    m = wt.fwt97_2d(m, LEVELS)
    print(m.min(), m.max())

    # Save the transformed image.
    #im = array_to_image(255 * (m - m.min()) / (m.max() - m.min()))
    im = array_to_image(m)
    im.save("test1_512_fwt.png")

    # Perform an inverse transform:
    m = wt.iwt97_2d(m, LEVELS)
    print(m.min(), m.max())

    # Save the inverse transformation.
    im = array_to_image(m)
    im.save("test1_512_iwt.png")
