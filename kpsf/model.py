#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["get_image"]

import numpy as np
try:
    import matplotlib.pyplot as pl
    pl = pl
except ImportError:
    pl = None

from ._kpsf import solve


def reshape_image(flux, ferr):
    w, h = flux.shape
    y, x = np.meshgrid(range(h), range(w))
    coords = np.vstack([x.flatten(), y.flatten(), flux.flatten(),
                        1.0 / ferr.flatten()]).T
    mask = np.prod(np.isfinite(coords[:, 2:]), axis=1)
    return coords, mask


class KeplerQuarter(object):

    def __init__(self, time, flux, ferr):
        # Save the raw data entries.
        self.time = time
        self.flux = flux
        self.ferr = ferr

        # Compute the times where the data is valid (i.e. not all NaNs).
        self.epochs = np.array([not np.all(np.isnan(f))
                                for f in self.flux], dtype=bool)
        self.nepochs = np.sum(self.epochs)

    def fit(self, psfpars, ff=None):
        # Coerce the data into the correct format for the solver.
        data, mask = zip(*[reshape_image(f, e)
                           for f, e in zip(self.flux[self.epochs],
                                           self.ferr[self.epochs])])
        data = np.array(data, dtype=float)
        mask = np.array(mask, dtype=bool)

        # Compute the pixels that are never NaNs.
        data = data[:, np.sum(mask, axis=0) > 0, :]

        # Deal with any left over missing data.
        inds = np.array(np.prod(np.isnan(data[:, :, 2:]), axis=2), dtype=bool)
        data[inds, 2:] = 0.0

        # Compute the median flux and normalize the data.
        norm = np.median(data[:, :, 2])
        data[:, :, 2:] /= norm

        # Estimate the centroids of the images.
        centroids = np.sum(data[:, :, :2] * data[:, :, 2][:, :, None],
                           axis=1) / np.sum(data[:, :, 2], axis=1)[:, None]

        # Estimate the flux of the star in each image.
        fluxes = np.sum(data[:, :, 2], axis=1)

        # Set up the initial conditions for the fit.
        initial = np.hstack((centroids, np.atleast_2d(fluxes).T))
        bias = np.zeros(data.shape[1])
        if ff is None:
            ff = np.ones(data.shape[1])

        # Plot the initial guess.
        fig = self.plot_model(data[0], initial[0], PSF(psfpars), ff)
        fig.savefig("initial.png")

        # Do the fit.
        N = len(data)
        info, final, ff, bias, psfpars = solve(data[:N], (0, 0),
                                               initial[:N],
                                               ff,
                                               bias,
                                               np.array(psfpars))

        # Plot the final model.
        fig = self.plot_model(data[0], final[0], PSF(psfpars), ff)
        fig.savefig("final.png")

    def plot_model(self, data, coords, psf, ff):
        # Compute the model given the parameters.
        x, y = np.array(data[:, 0], dtype=int), np.array(data[:, 1], dtype=int)
        d1, d2 = x - coords[0], y - coords[1]
        model = np.zeros((np.max(x)+1, np.max(y)+1))
        model[x, y] = ff * coords[2] * psf(d1, d2)

        # Unravel the data.
        img = np.zeros((np.max(x)+1, np.max(y)+1))
        img[x, y] = data[:, 2]

        fig = pl.figure(figsize=(10, 10))

        ax = fig.add_subplot(221)
        pl.imshow(img, cmap="gray", interpolation="nearest")
        pl.colorbar()
        ax.set_title("data")

        ax = fig.add_subplot(222)
        pl.imshow(model, cmap="gray", interpolation="nearest")
        pl.colorbar()
        ax.set_title("model")

        ax = fig.add_subplot(223)
        pl.imshow(img - model, cmap="gray", interpolation="nearest")
        pl.colorbar()
        ax.set_title("data - model")

        return fig


class PSF(object):

    def __init__(self, pars, ngaussians=2):
        self._ngaussians = ngaussians
        self._pars = pars

        self._det = np.zeros(ngaussians)
        self._invdet = np.zeros(ngaussians)
        self._factor = np.zeros(ngaussians)
        norm = 1.0
        for k in range(ngaussians):
            ind = 4 * k - 1
            self._det[k] = pars[ind+1]*pars[ind+3] - pars[ind+2]*pars[ind+2]
            self._invdet[k] = 1.0 / self._det[k]
            self._factor[k] = 0.5 * np.sqrt(self._invdet[k])
            if ind > 0:
                self._factor *= pars[ind]
                norm += pars[ind]
        self._factor /= norm * np.pi

    def __call__(self, d1, d2):
        value = np.zeros_like(d1)
        for k in range(self._ngaussians):
            ind = 4 * k - 1
            value += self._factor[k] * np.exp(-0.5 * self._invdet[k] *
                                              (self._pars[ind+3]*d1*d1
                                               + self._pars[ind+1]*d2*d2
                                               - 2*self._pars[ind+2]*d1*d2))
        print(np.sum(value))
        return value


def get_image(shape, pixels, coords, flat_field, bias, psfpars):
    psf = PSF(psfpars)
    pixels = np.array(pixels, dtype=int)
    dx, dy = pixels[:, 0] - coords[0], pixels[:, 1] - coords[1]
    img = np.zeros(shape)
    img[pixels[:, 0], pixels[:, 1]] = bias+flat_field*coords[2]*psf(dx, dy)
    return img
