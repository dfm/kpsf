#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

__all__ = ["reshape_image"]

import numpy as np


def reshape_image(flux, ferr):
    w, h = flux.shape
    y, x = np.meshgrid(range(h), range(w))
    coords = np.vstack([x.flatten(), y.flatten(), flux.flatten(),
                        1.0 / ferr.flatten()]).T
    mask = np.array(np.prod(np.isfinite(coords[:, 2:]), axis=1), dtype=bool)
    return coords[mask, :]
