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
    import matplotlib.pyplot as pl
    w, h = 9, 6
    psfpars = np.array([3.5, 0.1, 1.8])
    psf = PSF(psfpars)

    mask = 3 * np.ones((w, h), dtype=int)
    mask[[0, -1], 0] = 0
    mask[[0, -1], -1] = 0

    coords = np.array([3.56, 2.540, 10.])

    N = 20
    truth = np.empty([N, 3])
    data = []
    for i in range(N):
        coords += 0.1 * np.random.randn(3)
        truth[i] = coords
        data.append(generate_image(mask, truth[i], psf, 0.005))

    blah = np.zeros_like(mask, dtype=float)
    blah[np.array(data[-1][:, 0], dtype=int),
         np.array(data[-1][:, 1], dtype=int)] = data[-1][:, 2]
    pl.imshow(blah, cmap="gray", interpolation="nearest")
    pl.savefig("data.png")

    psfpars += 0.2 * np.random.randn(3)

    initial = np.array(truth)
    initial[:, 0] += 0.1 * np.random.randn(N)
    initial[:, 1] += 0.1 * np.random.randn(N)
    initial[:, 2] = 10 * np.ones(N)  # np.random.randn(N)

    info, coords, ff, psfpars = solve(data, [w, h], initial,
                                      np.ones(np.sum(mask)/3.0), psfpars)

    print(ff)

    pl.clf()
    pl.plot(truth[:, 2] / np.median(truth[:, 2]))
    pl.plot(coords[:, 2] / np.median(coords[:, 2]))
    pl.savefig("test.png")
