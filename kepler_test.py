#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import kplr
import kpsf
from kpsf._kpsf import solve
import numpy as np
import matplotlib.pyplot as pl

client = kplr.API()

# Load the data.
star = client.star(10592770)
pixels = star.get_target_pixel_files()[10]
lc = star.get_light_curves()[10].read()
table = pixels.read()

data = np.array([kpsf.reshape_image(flux, ferr)
                 for flux, ferr in zip(table["FLUX"], table["FLUX_ERR"])])
data = np.array([d for d in data if len(d)])

data[:, :, 2:] /= np.median(data[:, :, 2])

com = np.sum(data[:, :, :2] * data[:, :, 2][:, :, None], axis=1) \
    / np.sum(data[:, :, 2], axis=1)[:, None]

coords = np.hstack((com, np.atleast_2d(np.sum(data[:, :, 2], axis=1)).T))
psfpars = np.array([0.5, 0.0, 0.5])
flat_field = np.ones(len(data[0]))
bias = np.zeros(len(data[0]))

# Plot initial model.
img = table["FLUX"][0]
model = kpsf.get_image(img.shape, data[0, :, :2],
                       coords[0], flat_field, bias, psfpars)

pl.figure(figsize=(8, 10))
pl.subplot(211)
pl.imshow(img, cmap="gray", interpolation="nearest")
pl.plot(com[0, 1], com[0, 0], ".r")

pl.subplot(212)
pl.imshow(model, cmap="gray", interpolation="nearest")
pl.savefig("data.png")

# Do the solve.
N = 100
info, coords, ff, psfpars = solve(data[:N], img.shape, coords[:N],
                                  flat_field, bias, psfpars)

pl.clf()
f = coords[:, 2]
pl.plot(np.arange(N), f / np.median(f), ".k")
f = lc["SAP_FLUX"]
f = f[np.isfinite(f)][:N]
pl.plot(np.arange(N), f / np.median(f), ".r")
pl.savefig("flux.png")
