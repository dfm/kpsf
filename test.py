#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import numpy as np
from kpsf._kpsf import solve

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
    coords = np.vstack([x.flatten(), y.flatten(), img.flatten(),
                        np.ones_like(img).flatten() / noise]).T
    return coords[mask.flatten() > 0, :]


if __name__ == "__main__":
    w, h = 9, 6
    psfpars = np.array([3.5, 0.1, 1.8])
    psf = PSF(psfpars)

    mask = 3 * np.ones((w, h), dtype=int)
    mask[[0, -1], 0] = 0
    mask[[0, -1], -1] = 0

    coords = np.array([3.56, 1.540, 1.])
    data = generate_image(mask, coords, psf, 4e-3)

    print(coords, psfpars)

    coords += 0.2 * np.random.randn(3)
    psfpars += 0.2 * np.random.randn(3)

    coords = np.atleast_2d(coords)

    print(coords, psfpars)
    print(solve([data], [w, h], coords, psfpars))
