# -*- coding: utf-8 -*-

__version__ = "0.0.1"

try:
    __KPSF_SETUP__
except NameError:
    __KPSF_SETUP__ = False

if not __KPSF_SETUP__:
    __all__ = ["Frame", "TimeSeries", "PSF"]

    from .psf import PSF
    from .frame import Frame
    from .time_series import TimeSeries
