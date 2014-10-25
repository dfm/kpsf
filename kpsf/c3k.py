# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["find_centroid"]

import logging
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.ndimage.filters import gaussian_filter


# Build 3x3 matrix.
x, y = np.meshgrid(range(-1, 2), range(-1, 2), indexing="ij")
x, y = x.flatten(), y.flatten()
AT = np.vstack((x*x, y*y, x*y, x, y, np.ones_like(x)))
ATA = np.dot(AT, AT.T)
factor = cho_factor(ATA, overwrite_a=True)

# Build 5x5 matrix.
x, y = np.meshgrid(range(-2, 3), range(-2, 3), indexing="ij")
x, y = x.flatten(), y.flatten()
AT5 = np.vstack((x*x, y*y, x*y, x, y, np.ones_like(x)))
ATA5 = np.dot(AT5, AT5.T)
factor5 = cho_factor(ATA5, overwrite_a=True)


def fit_3x3(img):
    a, b, c, d, e, f = cho_solve(factor, np.dot(AT, img.flatten()))
    m = 1. / (4 * a * b - c*c)
    x = (c * e - 2 * b * d) * m
    y = (c * d - 2 * a * e) * m
    return x, y


def fit_5x5(img):
    a, b, c, d, e, f = cho_solve(factor5, np.dot(AT5, img.flatten()))
    m = 1. / (4 * a * b - c*c)
    x = (c * e - 2 * b * d) * m
    y = (c * d - 2 * a * e) * m
    return x, y


def find_centroid(img, smooth=-1, check_nan=True, fill_nan=True):
    if check_nan and not np.any(np.isfinite(img)):
        return np.nan, np.nan

    if fill_nan:
        m = np.isfinite(img)
        img[~m] = np.median(img[m])

    if smooth > 0:
        img = gaussian_filter(img, smooth, mode="nearest")

    xi, yi = np.unravel_index(np.argmax(img), img.shape)
    if not (xi >= 1 and xi < img.shape[0]-1 and
            yi >= 1 and yi < img.shape[1]-1):
        logging.warn("Maximum pixel is at the edge.")
        return np.nan, np.nan

    ox, oy = fit_3x3(img[xi-1:xi+2, yi-1:yi+2])

    return xi + ox, yi + ox


def test():
    import matplotlib.pyplot as pl

    # Generate a synthetic image.
    cx, cy = 2.389, 6.403
    x, y = np.meshgrid(range(10), range(13), indexing="ij")
    r2 = (x - cx) ** 2 + (y - cy) ** 2
    img = np.exp(-0.5 * r2 / 1.5 ** 2)

    x0, y0 = find_centroid(img)
    print(x0 - cx, y0 - cy)

    pl.pcolor(x, y, img, cmap="gray")
    pl.plot(cx + 0.5, cy + 0.5, "+r")
    # pl.plot(x0 + 0.5, y0 + 0.5, "+b")

    pl.savefig("test.png")


if __name__ == "__main__":
    test()
