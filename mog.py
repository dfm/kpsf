#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = []

import numpy as np


def eval_gaussian(x, y, params):
    det = params[3] * params[5] - params[4] * params[4]
    dx = x - params[1]
    dy = y - params[2]
    v = dx*(params[5]*dx-params[4]*dy) + dy*(params[3]*dy-params[4]*dx)
    return params[0] * np.exp(-0.5 * v / det) / np.sqrt(det) / (2 * np.pi)


if __name__ == "__main__":
    import os
    import fitsio
    import matplotlib.pyplot as pl

    with fitsio.FITS("data/kplr07.4_2011265_prf.mog.fits") as f:
        for i, hdu in enumerate(range(1, len(f))):
            data = fitsio.read("data/kplr07.4_2011265_prf.fits", hdu)
            params = f[hdu].read()

            x, y = np.meshgrid(np.arange(550)/50., np.arange(550)/50.)
            img = np.zeros((550, 550))
            for p in params:
                print(p)
                img += eval_gaussian(x-5.5, y-5.5, p)

            d = "mog/{0}".format(i)
            try:
                os.makedirs(d)
            except os.error:
                pass

            pl.clf()
            vmin, vmax = 0.08, 0.2
            pl.imshow(data, cmap="gray", interpolation="nearest", vmin=vmin,
                      vmax=vmax)
            pl.xlim(0, 550)
            pl.ylim(0, 550)
            pl.colorbar()
            pl.savefig("{0}/prf.png".format(d))

            pl.clf()
            pl.imshow(img, cmap="gray", interpolation="nearest", vmin=vmin,
                      vmax=vmax)
            pl.plot(params["xpos"]*50+275, params["ypos"]*50+275, "+r")
            pl.xlim(0, 550)
            pl.ylim(0, 550)
            pl.colorbar()
            pl.savefig("{0}/mog.png".format(d))

            pl.clf()
            pl.imshow(img - data, cmap="gray", interpolation="nearest")
            pl.xlim(0, 550)
            pl.ylim(0, 550)
            pl.colorbar()
            pl.savefig("{0}/delta.png".format(d))

    assert 0

    data = fitsio.read("data/kplr07.4_2011265_prf.fits", 2)

    x, y = np.meshgrid(range(550), range(550))
    img = np.zeros((550, 550))
    for i in range(int(len(params) / 6)):
        img += eval_gaussian(x, y, params[6*i:6*(i+1)])

    d = "demo/{0}".format(int(len(params)/6))
    try:
        os.makedirs(d)
    except os.error:
        pass

    pl.clf()
    vmin, vmax = 0.05, 0.2
    pl.imshow(data, cmap="gray", interpolation="nearest", vmin=vmin, vmax=vmax)
    pl.xlim(0, 550)
    pl.ylim(0, 550)
    pl.colorbar()
    pl.savefig("{0}/mog_test_prf.png".format(d))

    pl.clf()
    pl.imshow(img, cmap="gray", interpolation="nearest", vmin=vmin, vmax=vmax)
    pl.plot(params[1::6], params[2::6], "+r")
    pl.xlim(0, 550)
    pl.ylim(0, 550)
    pl.colorbar()
    pl.savefig("{0}/mog_test_mog.png".format(d))

    pl.clf()
    pl.imshow(img - data, cmap="gray", interpolation="nearest")
    pl.xlim(0, 550)
    pl.ylim(0, 550)
    pl.colorbar()
    pl.savefig("{0}/mog_test_delta.png".format(d))
