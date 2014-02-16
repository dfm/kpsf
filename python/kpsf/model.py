#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["MOMOG"]

import os
import fitsio
import numpy as np


class Gaussian(object):

    def __init__(self, amp, xpos, ypos, xvar, covar, yvar):
        self.xpos = xpos
        self.ypos = ypos
        self.xvar = xvar
        self.covar = covar
        self.yvar = yvar

        self.det = xvar * yvar - covar * covar
        self.factor = amp / (2*np.pi*np.sqrt(self.det))

    def __call__(self, i, j):
        dx = self.xpos - i
        dy = self.ypos - j
        x = self.yvar * dx - self.covar * dy
        y = self.xvar * dy - self.covar * dx
        v = (dx * x + dy * y) / self.det
        return self.factor * np.exp(-0.5 * v)


class MOG(object):

    def __init__(self, *args):
        self.gaussians = [Gaussian(*a) for a in zip(*args)]

    def __call__(self, i, j):
        i = np.atleast_1d(i)
        ret = np.zeros(i.shape)
        for g in self.gaussians:
            ret += g(i, j)
        return ret


class MOMOG(object):

    def __init__(self, fn):
        if not os.path.exists(fn):
            raise RuntimeError("MOG file {0} doesn't exist".format(fn))

        # Load the PSF basis Gaussians.
        self.mogs = [MOG(*zip(*(fitsio.read(fn, i+1)))) for i in range(5)]

    def __call__(self, coeffs, i, j):
        i = np.atleast_1d(i)
        ret = np.zeros(i.shape)
        for k, g in enumerate(self.mogs):
            ret += coeffs[k] * g(i, j)
        return ret
