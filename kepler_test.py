#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import time
import kplr
import kpsf
from kpsf._kpsf import solve
from kpsf.model import KeplerQuarter
import numpy as np
import matplotlib.pyplot as pl

client = kplr.API()

# Load the data.
star = client.star(10592770)
pixels = star.get_target_pixel_files()[10]
lc = star.get_light_curves()[10].read()
table = pixels.read()

model = KeplerQuarter(table["TIME"], table["FLUX"], table["FLUX_ERR"])
model.fit([0.5, 0.0, 0.5, 0.01, 5.0, 0.0, 5.0])

assert 0

data = np.array([kpsf.reshape_image(flux, ferr)
                 for flux, ferr in zip(table["FLUX"], table["FLUX_ERR"])])
data = np.array([d for d in data if len(d)])

data[:, :, 2:] /= np.median(data[:, :, 2])

com = np.sum(data[:, :, :2] * data[:, :, 2][:, :, None], axis=1) \
    / np.sum(data[:, :, 2], axis=1)[:, None]

coords = np.hstack((com, np.atleast_2d(np.sum(data[:, :, 2], axis=1)).T))
psfpars = np.array([0.5, 0.0, 0.5, 0.0, 5.0, 0.0, 5.0])
flat_field = np.ones(len(data[0]))
bias = np.zeros(len(data[0]))

# Plot initial model.
img = table["FLUX"][0]
model = kpsf.get_image(img.shape, data[0, :, :2],
                       coords[0], flat_field, bias, psfpars)

pl.figure(figsize=(8, 10))
pl.subplot(311)
pl.imshow(img, cmap="gray", interpolation="nearest")
pl.plot(com[0, 1], com[0, 0], ".r")

pl.subplot(312)
pl.imshow(model, cmap="gray", interpolation="nearest")

pl.subplot(313)
pl.imshow(model - img, cmap="gray", interpolation="nearest")
pl.savefig("data.png")

# Do the solve.
N = len(data)
print(N)
strt = time.time()
info, coords, ff, bias, psfpars = solve(data[:N], img.shape, coords[:N],
                                        flat_field, bias, psfpars)
print("Took {0} seconds".format(time.time() - strt))
print(psfpars)

pl.clf()
f = coords[:, 2]
pl.plot(np.arange(N), f / np.median(f), ".k")
f = lc["SAP_FLUX"]
f = f[np.isfinite(f)][:N]
# pl.plot(np.arange(N), f / np.median(f), ".r")
pl.savefig("flux.png")


# plot fit.
pl.clf()
pl.figure(figsize=(8, 10))
pl.subplot(311)
pl.imshow(img, cmap="gray", interpolation="nearest")
pl.plot(com[0, 1], com[0, 0], ".r")
pl.colorbar()

model = kpsf.get_image(img.shape, data[0, :, :2],
                       coords[0], flat_field, bias, psfpars)
pl.subplot(312)
pl.imshow(model, cmap="gray", interpolation="nearest")
pl.colorbar()

pl.subplot(313)
pl.imshow(model - img, cmap="gray", interpolation="nearest")
pl.colorbar()
pl.savefig("final.png")
