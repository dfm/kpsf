#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = []

import fitsio
import numpy as np
import matplotlib.pyplot as pl

from kpsf._kpsf import (compute_model, run_photometry_all,
                        N_INT_TIME, N_PSF_COMP)

# data = fitsio.read("data/ktwo202065500-c00_lpd-targ.fits.gz")
data = fitsio.read("data/kplr060021426-2014044044430_lpd-targ.fits")
time = data["TIME"]
flux = np.array(data["RAW_CNTS"], dtype=np.float64)
ferr = np.sqrt(flux)
# flux = data["FLUX"]
# ferr = data["FLUX_ERR"]

x0, y0 = np.mean([np.unravel_index(np.argmax(f), f.shape) for f in flux],
                 axis=0)
x0, y0 = map(int, map(np.round, [x0, y0]))
print(x0, y0)
d = 5

flux = flux[:, x0-d:x0+d+1, y0-d:y0+d+1]
ferr = ferr[:, x0-d:x0+d+1, y0-d:y0+d+1]

# mu = np.median(flux)
# print(float(mu))
# flux = flux / mu
# ferr = ferr / mu

xpix, ypix = np.meshgrid(range(flux.shape[1]), range(flux.shape[2]),
                         indexing="ij")
xpix = np.ascontiguousarray(xpix.flatten(), dtype=np.float64)
ypix = np.ascontiguousarray(ypix.flatten(), dtype=np.float64)
shape = flux[0].shape

flux = flux.reshape((len(flux), -1))
ferr = ferr.reshape((len(ferr), -1))

m = (np.sum(flux, axis=1) > 0.0)
# m = (np.sum(flux, axis=1) > 0.0) * (np.arange(len(flux)) % 3 == 0)
# m = (np.sum(flux, axis=1) > 0.0) * (np.arange(len(flux)) < 100)
time = np.ascontiguousarray(time[m], dtype=np.float64)
flux = np.ascontiguousarray(flux[m], dtype=np.float64)
ferr = np.ascontiguousarray(ferr[m], dtype=np.float64)

max_fracs = np.array([1.0] * (N_PSF_COMP - 1))


def fit_one(flux, ferr, bg, model, coords, coeffs):
    f2 = (flux - np.median(flux)) ** 2
    x0 = np.sum(f2 * xpix) / np.sum(f2)
    y0 = np.sum(f2 * ypix) / np.sum(f2)
    # i = np.argmax(flux)
    # x0, y0 = xpix[i], ypix[i]

    # Initialize the parameters.
    coords[:] = np.array([x0, y0] * N_INT_TIME)
    coords[:] += 1e-8 * np.random.randn(len(coords))
    coeffs[:] = ([1.0, 1.0, 0.0]
                 + [v for j in range(N_PSF_COMP-1)
                    for v in [-100.0, 0.0, 0.0, 5.0+j, 5.0+j, 0.0]])

    # Do the initial least squares fit.
    m = compute_model(max_fracs, xpix, ypix, 1.0, coords, coeffs,
                      np.ones((len(xpix),), dtype=np.float64), 0.0)
    A = np.vander(m, 2)
    w = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, flux))
    model[0] = w[0]
    bg[0] = w[1]


bg = np.zeros(len(flux), dtype=np.float64)
model = np.zeros(len(flux), dtype=np.float64)
coords = np.zeros((len(flux), 2 * N_INT_TIME), dtype=np.float64)
coeffs = np.zeros((len(flux), 6*N_PSF_COMP - 3), dtype=np.float64)
for i in range(len(flux)):
    if not np.any(np.isfinite(flux[i])):
        continue
    fit_one(flux[i], ferr[i], bg[i:i+1], model[i:i+1], coords[i, :],
            coeffs[i, :])

ff = np.ones(len(xpix), dtype=np.float64)
coeffs = np.ascontiguousarray(np.median(coeffs, axis=0), dtype=np.float64)
run_photometry_all(time, xpix, ypix, flux, ferr, model, coords, coeffs, ff, bg,
                   max_fracs, 0.1, 1e-10)
# assert 0

# print(coords)
# print(coeffs)

mu = np.median(model)
print(np.sqrt(np.median((model - mu) ** 2)) / mu)

pl.clf()
pl.plot(time, model, ".k")
# pl.plot(time, coords[:, 2], ".k")
pl.plot(time, bg, ".r")
pl.savefig("dude.png")
assert 0

fig = pl.figure(figsize=(10, 10))

i = 0
vmin, vmax = np.min(flux), np.max(flux)
for n in range(len(flux)):
    if not np.any(np.isfinite(flux[i])):
        continue

    pl.clf()
    ax = fig.add_subplot(221)
    ax.imshow(flux[n].reshape(shape), cmap="gray", interpolation="nearest",
              vmin=vmin, vmax=vmax)
    ax.plot(coords[n, 1::2], coords[n, 0::2], "r")
    ax.plot(coords[n, 1::2], coords[n, 0::2], "+r")
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax = fig.add_subplot(222)
    m = compute_model(max_fracs, xpix, ypix, model[n], coords[n], coeffs, ff,
                      bg[n])
    ax.imshow(m.reshape(shape), cmap="gray",
              interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.plot(coords[n, 1::2], coords[n, 0::2], "r")
    ax.plot(coords[n, 1::2], coords[n, 0::2], "+r")
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax = fig.add_subplot(223)
    d = vmax - vmin
    ax.imshow((flux[n] - m).reshape(shape), cmap="gray",
              interpolation="nearest", vmin=-0.5*d, vmax=0.5*d)
    ax.plot(coords[n, 1::2], coords[n, 0::2], "r")
    ax.plot(coords[n, 1::2], coords[n, 0::2], "+r")
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax = fig.add_subplot(224)
    ax.imshow(ff.reshape(shape), cmap="gray",
              interpolation="nearest", vmin=0.9, vmax=1.1)
    ax.plot(coords[n, 1::2], coords[n, 0::2], "r")
    ax.plot(coords[n, 1::2], coords[n, 0::2], "+r")
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    pl.savefig("frames/{0:05d}.png".format(i))
    i += 1
    print(i)
