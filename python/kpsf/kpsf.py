#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = []

try:
    import kplr
    kplr = kplr
except ImportError:
    kplr = None

import os
import numpy as np
import matplotlib.pyplot as pl

from kpsf.model import MOMOG
from kpsf._kpsf import photometry


def get_models(kicid, quarter, prf_path, ttol=0.1,
               fn_template="kplr{0:02d}.{1}_2011265_prf.mog.fits".format):
    if kplr is None:
        raise ImportError("kplr is required")

    client = kplr.API()
    tpf = client.target_pixel_files(kicid, sci_data_quarter=quarter)[0]
    data = tpf.read()
    mask = np.array(tpf.read(ext=2), dtype=bool)

    # Build the filename for the MOG file.
    fn = os.path.join(prf_path, fn_template(tpf.sci_module, tpf.sci_output))

    # Read in the data.
    time = np.array(data["TIME"], dtype=float)
    q = np.array(data["QUALITY"], dtype=int)
    flux = np.array(data["RAW_CNTS"], dtype=float)
    ferr = np.empty_like(flux)
    ferr[flux > 0] = np.sqrt(flux[flux > 0])
    # flux = np.array(data["FLUX"], dtype=float)
    # ferr = np.array(data["FLUX_ERR"], dtype=float)

    # Flatten the data into the correct form.
    nx, ny = mask.shape
    y, x = np.meshgrid(range(ny), range(nx))
    mask = mask.flatten()
    flux = flux.reshape((-1, nx*ny))[:, mask]
    ferr = ferr.reshape((-1, nx*ny))[:, mask]
    x = np.array(x.flatten()[mask], dtype=np.int32)
    y = np.array(y.flatten()[mask], dtype=np.int32)

    # Mask the NaN-ed times.
    m = np.all(np.isfinite(flux), axis=1) * np.isfinite(time)
    time = time[m]
    flux = flux[m]
    ferr = ferr[m]
    q = q[m]

    # Compute the acceptable bitmask.
    bitmask = (q & (1 | 2 | 4 | 8 | 16 | 32 | 64 | 256 | 4096)) == 0

    # Cut the dataset on breaks in time.
    times = time[:-1][time[1:]-time[:-1] > ttol]

    # Loop over sections.
    models = []
    for i1, i2 in zip(np.append(0, times), np.append(times, np.inf)):
        inds = (i1 <= time) * (time < i2) * bitmask
        models.append(KPSF(fn, time[inds], x, y, flux[inds],
                           ferr[inds], q[inds]))

    return models


class KPSF(object):

    def __init__(self, basis_fn, time, x, y, flux, ferr, quality):
        self.basis_fn = basis_fn
        self.time = time
        self.x = x
        self.y = y
        self.flux = flux
        self.ferr = ferr
        self.quality = quality

        # Load the PSF basis.
        self.basis = MOMOG(basis_fn)

        # Initialize some of the parameters.
        self.coeffs = np.ones(5) / 5.0
        self.flat = np.ones_like(self.flux[0])

        # Initialize coordinates, flux, and background.
        self.initialize()

    def initialize(self):
        # Estimate the centroid.
        norm = np.sum(self.flux, axis=1)
        cx = np.sum(self.flux * self.x[None, :], axis=1) / norm
        cy = np.sum(self.flux * self.y[None, :], axis=1) / norm

        # Estimate the flux/background for each exposure.
        psf = self.basis(self.coeffs, self.x[None, :] - cx[:, None],
                         self.y[None, :] - cy[:, None])
        f, b = np.empty_like(cx), np.empty_like(cx)
        for i, p in enumerate(psf):
            f[i], b[i] = np.linalg.lstsq(np.vstack((p, np.ones_like(p))).T,
                                         self.flux[i])[0]

        # Initialize the background level and flux values based on these
        # estimates.
        self.coords = np.vstack((f, cx, cy)).T
        self.background = b

    def fit(self, maxiter=1000):
        photometry(maxiter, self.basis_fn, 10000., 0., 0., 1e3, self.x, self.y,
                   self.flux, self.ferr, self.coeffs, self.coords, self.flat,
                   self.background)
