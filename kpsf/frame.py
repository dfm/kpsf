# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["Frame"]

import numpy as np
from simplexy import simplexy

from ._kpsf import compute_psf, compute_scene


class Frame(object):

    def __init__(self, img, err_img, mask=None, **kwargs):
        if mask is None:
            self.mask = np.isfinite(img)
        else:
            self.mask = mask
        self.img = img

        # Compute the pixel positions.
        self.shape = self.img.shape
        x, y = np.meshgrid(range(self.shape[0]), range(self.shape[1]),
                           indexing="ij")
        self.xi = np.array(x[self.mask], dtype=np.int)
        self.x = np.array(self.xi, dtype=np.float64)
        self.yi = np.array(y[self.mask], dtype=np.int)
        self.y = np.array(self.yi, dtype=np.float64)

        # Initialize the coordinate set.
        self.initialize(**kwargs)

    def initialize(self, **kwargs):
        if not np.any(self.mask):
            self.coords = []
            return
        tmp = np.array(self.img)
        tmp[~(self.mask)] = np.median(tmp[self.mask])
        self.coords = simplexy(tmp, **kwargs)

    def __len__(self):
        return len(self.coords) if self.coords is not None else 0

    def predict(self, psf, origin=None, offsets=None, fluxes=None,
                background=None, response=None):
        # Assign the default parameters.
        if origin is None:
            origin = np.array([0.0, 0.0])
        if offsets is None:
            offsets = np.vstack((self.coords["x"], self.coords["y"])).T
            offsets -= origin[None, :]
        if fluxes is None:
            norm = compute_psf(np.ascontiguousarray(psf.pars, np.float64),
                               0.0, 0.0)
            fluxes = np.array(self.coords["flux"] / norm, dtype=np.float64)
        if background is None:
            background = np.median(self.coords["bkg"])
        if response is None:
            response = np.ones(self.shape)

        # Compute the prediction.
        tmp = compute_scene(self.x, self.y,
                            np.ascontiguousarray(fluxes, np.float64),
                            np.ascontiguousarray(origin, np.float64),
                            np.ascontiguousarray(offsets, np.float64),
                            np.ascontiguousarray(psf.pars, np.float64),
                            float(background),
                            np.ascontiguousarray(response[self.mask],
                                                 np.float64))

        # Reshape the prediction.
        img = np.nan + np.zeros(self.shape)
        img[self.xi, self.yi] = tmp
        return img
