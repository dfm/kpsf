#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = []

import pickle
import numpy as np
import matplotlib.pyplot as pl

from kpsf._kpsf import N_STARS, N_PSF_COMP, run_photometry_all

x0, a, W, time, flux, ferr = pickle.load(open("blugh.pkl", "r"))

# Flatten the flux images into vectors.
xpix, ypix = np.meshgrid(range(flux.shape[1]), range(flux.shape[2]),
                         indexing="ij")
xpix = np.ascontiguousarray(xpix.flatten(), dtype=np.float64)
ypix = np.ascontiguousarray(ypix.flatten(), dtype=np.float64)
shape = flux[0].shape

flux = flux.reshape((len(flux), -1))
ferr = ferr.reshape((len(ferr), -1))

m = (np.sum(flux, axis=1) > 0.0)
time = np.ascontiguousarray(time[m], dtype=np.float64)
flux = np.ascontiguousarray(flux[m], dtype=np.float64)
ferr = np.ascontiguousarray(ferr[m], dtype=np.float64)
W = np.ascontiguousarray(W[m], dtype=np.float64)
a = np.ascontiguousarray(a.T, dtype=np.float64)

# Remove missing pixels.
m = np.all(ferr > 0.0, axis=0)
flux = np.ascontiguousarray(flux[:, m], dtype=np.float64)
ferr = np.ascontiguousarray(ferr[:, m], dtype=np.float64)
xpix = xpix[m]
ypix = ypix[m]

bg = np.ascontiguousarray(np.median(flux, axis=1), dtype=np.float64)
model = np.ascontiguousarray(np.array([19748.9, 1229.04, 749.369])[None, :]
                             + np.zeros((len(flux), N_STARS)),
                             dtype=np.float64)
psfpars = np.array([1.0, 1.0, 0.0]
                   + [v for j in range(N_PSF_COMP-1)
                      for v in [-100.0, 0.0, 0.0, 5.0+j, 5.0+j, 0.0]],
                   dtype=np.float64)
ff = np.ones(len(xpix), dtype=np.float64)
max_fracs = np.array([1.0] * (N_PSF_COMP - 1))

n = len(W)
run_photometry_all(W[:n], time[:n], xpix, ypix, flux[:n], ferr[:n],
                   model[:n], x0, a, psfpars,
                   ff, bg[:n], max_fracs, 1.0, 1e-5)

print(model)
