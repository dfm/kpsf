# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["PSF"]

import numpy as np

from ._kpsf import compute_psf


class PSF(object):

    def __init__(self, var_x, var_y, var_xy=0.0):
        self.var_x = var_x
        self.var_y = var_y
        self.var_xy = var_xy
        self.components = []

    def add_component(self, amp, var_x, var_y, var_xy=0.0, dx=0.0, dy=0.0):
        self.components.append([amp, dx, dy, var_x, var_y, var_xy])

    @property
    def pars(self):
        return np.array([self.var_x, self.var_y, self.var_xy]
                        + [v for c in self.components for v in c],
                        dtype=np.float64)

    @pars.setter
    def pars(self, v):
        self.var_x, self.var_y, self.var_xy = v[:3]
        for i in range(len(self.components)):
            self.components[i] = v[3+6*i:9+6*i]

    def __call__(self, dx, dy):
        return compute_psf(self.pars, dx, dy)
