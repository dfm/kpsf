#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

__all__ = []

import fitsio
import numpy as np


class PSF(object):

    def __init__(self, pars):
        self.pars = pars
        self.invdet = 0.5 / (pars[0] * pars[2] - pars[1] * pars[1])
        self.factor = np.sqrt(self.invdet) / np.pi

    def __call__(self, dx, dy):
        return self.factor * np.exp(-self.invdet *
                                    (self.pars[2] * dx * dx +
                                     self.pars[0] * dy * dy -
                                     2 * self.pars[1] * dx * dy))


def generate_image(mask, coords, psf, noise):
    w, h = mask.shape
    y, x = np.meshgrid(range(h), range(w))
    dx, dy = x - coords[0], y - coords[1]
    img = coords[2] * psf(dx, dy) \
        + noise * np.random.randn(w*h).reshape((w, h))
    img[mask == 0] = np.nan
    return img, noise * np.ones_like(img)

if __name__ == "__main__":
    w, h = 9, 6
    psf = PSF([3.5, 0.1, 1.8])

    mask = 3 * np.ones((w, h), dtype=int)
    mask[[0, -1], 0] = 0
    mask[[0, -1], -1] = 0

    coords = np.array([3.56, 1.540, 1.])
    time = np.arange(400.0, 410.0, 1.0 / 24. / 2.)
    truth = []
    flux = []
    ferr = []

    for i, t in enumerate(time):
        coords += 0.01 * np.random.randn(len(coords))
        results = generate_image(mask, coords, psf, 4e-3)
        truth.append(np.array(coords))
        flux.append(results[0])
        ferr.append(results[1])

    data = np.zeros(len(time), dtype=[("time", "f4"),
                                      ("flux", "f4", mask.shape),
                                      ("ferr", "f4", mask.shape),
                                      ("truth", "f4", (3,))])
    data["time"] = time
    data["flux"] = flux
    data["ferr"] = ferr
    data["truth"] = truth

    fits = fitsio.FITS("data.fits", "rw")
    fits.write(data)
    fits.write(mask)
    fits.close()
