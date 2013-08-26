#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["get_image"]

import numpy as np


class PSF(object):

    def __init__(self, pars):
        self._pars = pars

        norm = 1.0 / (1 + pars[3])
        self._invdet1 = 0.5 / (pars[0] * pars[2] - pars[1] * pars[1])
        self._factor1 = norm * np.sqrt(self._invdet1) / np.pi

        self._invdet2 = 0.5 / (pars[4] * pars[6] - pars[5] * pars[5])
        self._factor2 = norm * pars[3] * np.sqrt(self._invdet2) / np.pi

    def __call__(self, dx, dy):
        return (self._factor1 * np.exp(-self._invdet1 *
                                       (self._pars[2] * dx * dx +
                                        self._pars[0] * dy * dy -
                                        2 * self._pars[1] * dx * dy))
                + self._factor2 * np.exp(-self._invdet2 *
                                         (self._pars[6] * dx * dx +
                                          self._pars[4] * dy * dy -
                                          2 * self._pars[5] * dx * dy)))


def get_image(shape, pixels, coords, flat_field, bias, psfpars):
    psf = PSF(psfpars)
    pixels = np.array(pixels, dtype=int)
    dx, dy = pixels[:, 0] - coords[0], pixels[:, 1] - coords[1]
    img = np.zeros(shape)
    img[pixels[:, 0], pixels[:, 1]] = bias+flat_field*coords[2]*psf(dx, dy)
    return img
