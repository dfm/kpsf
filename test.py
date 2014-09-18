#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = []

import fitsio
import numpy as np
import matplotlib.pyplot as pl

from kpsf._kpsf import compute_model, run_photometry_all


def centroid(x0, y0, img):
    x0, y0 = int(x0), int(y0)
    stamp = img[x0-1:x0+2, y0-1:y0+2]
    return stamp


# def psf(


data = fitsio.read("kplr060021426-2014044044430_lpd-targ.fits")
time = data["TIME"]
flux = data["FLUX"]
ferr = data["FLUX_ERR"]

x0, y0 = np.mean([np.unravel_index(np.argmax(f), f.shape) for f in flux],
                 axis=0)
x0, y0 = map(int, map(np.round, [x0, y0]))
print(x0, y0)
d = 7

flux = flux[:, x0-d:x0+d+1, y0-d:y0+d+1]
ferr = ferr[:, x0-d:x0+d+1, y0-d:y0+d+1]

mu = np.mean(flux)
flux /= mu
ferr /= mu

xpix, ypix = np.meshgrid(range(flux.shape[1]), range(flux.shape[2]),
                         indexing="ij")
xpix = np.ascontiguousarray(xpix.flatten(), dtype=np.float64)
ypix = np.ascontiguousarray(ypix.flatten(), dtype=np.float64)
shape = flux[0].shape

flux = flux.reshape((len(flux), -1))
ferr = ferr.reshape((len(ferr), -1))

m = np.sum(flux, axis=1) > 0.0
time = np.ascontiguousarray(time[m], dtype=np.float64)
flux = np.ascontiguousarray(flux[m], dtype=np.float64)
ferr = np.ascontiguousarray(ferr[m], dtype=np.float64)

print(np.sqrt(np.median((flux - np.median(flux))**2)))
print(np.median(flux))


def fit_one(flux, ferr, bg, coords, coeffs):
    # x0 = np.sum(xpix * flux) / np.sum(flux)
    # y0 = np.sum(ypix * flux) / np.sum(flux)
    i = np.argmax(flux)
    x0, y0 = xpix[i], ypix[i]

    # Initialize the parameters.
    coords[:] = [1.0, x0, y0]
    coeffs[:] = [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 5.0, 5.0, 0.0]

    # Do the initial least squares fit.
    model = compute_model(xpix, ypix, coords, coeffs,
                          np.ones((len(xpix),), dtype=np.float64),
                          np.zeros((1,), dtype=np.float64))
    A = np.vander(model, 2)
    w = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, flux))
    coords[0] = w[0]
    bg[0] = w[1]

    # Do the optimization.
    # run_photometry_one(xpix, ypix, flux, ferr, coords, coeffs, bg)


bg = np.zeros(len(flux), dtype=np.float64)
coords = np.zeros((len(flux), 3), dtype=np.float64)
coeffs = np.zeros((len(flux), 9), dtype=np.float64)
for i in range(len(flux)):
    if not np.any(np.isfinite(flux[i])):
        continue
    fit_one(flux[i], ferr[i], bg[i:i+1], coords[i, :], coeffs[i, :])

ff = np.ones(len(xpix), dtype=np.float64)
coeffs = np.ascontiguousarray(np.median(coeffs, axis=0), dtype=np.float64)
run_photometry_all(time, xpix, ypix, flux, ferr, coords, coeffs, ff, bg)

print(coords)
print(coeffs)

pl.plot(time, coords[:, 0], ".k")
# pl.plot(time, coords[:, 2], ".k")
pl.plot(time, 100 * bg, ".r")
pl.savefig("dude.png")

fig = pl.figure(figsize=(10, 10))

i = 0
for n in range(len(flux)):
    if not np.any(np.isfinite(flux[i])):
        continue

    pl.clf()

    vmin, vmax = np.min(flux), np.max(flux)
    ax = fig.add_subplot(221)
    ax.imshow(flux[n].reshape(shape), cmap="gray", interpolation="nearest",
              vmin=vmin, vmax=vmax)
    ax.plot(coords[n, 2], coords[n, 1], "+r")
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax = fig.add_subplot(222)
    model = compute_model(xpix, ypix, coords[n], coeffs, ff, bg[n:n+1])
    ax.imshow(model.reshape(shape), cmap="gray",
              interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.plot(coords[n, 2], coords[n, 1], "+r")
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax = fig.add_subplot(223)
    ax.imshow((flux[n] - model).reshape(shape), cmap="gray",
              interpolation="nearest")
    ax.plot(coords[n, 2], coords[n, 1], "+r")
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax = fig.add_subplot(224)
    ax.imshow(ff.reshape(shape), cmap="gray",
              interpolation="nearest")
    ax.plot(coords[n, 2], coords[n, 1], "+r")
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    pl.savefig("frames/{0:05d}.png".format(i))
    i += 1
