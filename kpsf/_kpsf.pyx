from __future__ import division

cimport cython

import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef extern from "constants.h":
    cdef int NUM_INT_TIME
    cdef int NUM_STARS
    cdef int NUM_PSF_COMP
N_STARS = NUM_STARS
N_INT_TIME = NUM_INT_TIME
N_PSF_COMP = NUM_PSF_COMP

cdef extern from "kpsf.h" namespace "kpsf":
    cdef int photometry_all (
        # INPUTS
        const int nw,         # The number of latent dimensions in the pointing model.
        const int nt,         # The number of time points.
        const int npix,       # The number of pixels.
        const double* w,      # The [nt, nw] list of latent parameters.
        const double* xpix,   # The [npix] list of pixel center coordinates.
        const double* ypix,   # ...
        const double* flux,   # The [nt, npix] list of measured pixel values.
        const double* ferr,   # The uncertainties on these values.

        # OUTPUTS
        double* model,        # The [nt, NUM_STARS] list of fluxes.
        double* x0,           # The [NUM_STARS, 2] set of mean star locations.
        double* a,            # The [nw, 2] "a" matrix.
        double* psfpars,      # The PSF parameters.
        double* ff,           # The [npix] list of pixel responses.
        double* bg,           # The [nt] list of background levels.

        # TUNING PARAMETERS
        const double* max_fracs, const double motion_reg, const double ff_reg
    )

cdef extern from "psf.h" namespace "kpsf":
    cdef int evaluate_psf[T] (const double* max_fracs,
                              const double x, const double y,
                              const T& x0, const T& y0, const T* params,
                              T* value)


def run_photometry_all(np.ndarray[DTYPE_t, ndim=2] w,
                       np.ndarray[DTYPE_t, ndim=1] time,
                       np.ndarray[DTYPE_t, ndim=1] xpix,
                       np.ndarray[DTYPE_t, ndim=1] ypix,
                       np.ndarray[DTYPE_t, ndim=2] flux,
                       np.ndarray[DTYPE_t, ndim=2] ferr,
                       np.ndarray[DTYPE_t, ndim=2] model,
                       np.ndarray[DTYPE_t, ndim=2] x0,
                       np.ndarray[DTYPE_t, ndim=2] a,
                       np.ndarray[DTYPE_t, ndim=1] psfpars,
                       np.ndarray[DTYPE_t, ndim=1] ff,
                       np.ndarray[DTYPE_t, ndim=1] bg,
                       np.ndarray[DTYPE_t, ndim=1] max_fracs,
                       double motion_reg, double ff_reg):
    cdef int nw = w.shape[1]
    cdef int nt = time.shape[0]
    cdef int npix = xpix.shape[0]
    if (w.shape[0] != nt or a.shape[0] != nw or a.shape[1] != 2
            or x0.shape[0] != NUM_STARS or x0.shape[1] != 2
            or npix != ypix.shape[0] or nt != flux.shape[0]
            or npix != flux.shape[1]
            or nt != ferr.shape[0] or npix != ferr.shape[1]
            or model.shape[0] != nt or model.shape[1] != NUM_STARS
            or psfpars.shape[0] != 6*NUM_PSF_COMP-3 or bg.shape[0] != nt
            or ff.shape[0] != npix or max_fracs.shape[0] != NUM_PSF_COMP-1):
        raise ValueError("Invalid dimensions")

    photometry_all(nw, nt, npix, <double*>w.data,
                   <double*>xpix.data, <double*>ypix.data,
                   <double*>flux.data, <double*>ferr.data,
                   <double*>model.data,
                   <double*>x0.data, <double*>a.data,
                   <double*>psfpars.data,
                   <double*>ff.data, <double*>bg.data,
                   <double*>max_fracs.data, motion_reg, ff_reg)


def compute_model(np.ndarray[DTYPE_t, ndim=1] max_fracs,
                  np.ndarray[DTYPE_t, ndim=1] xpix,
                  np.ndarray[DTYPE_t, ndim=1] ypix,
                  double model,
                  np.ndarray[DTYPE_t, ndim=1] coords,
                  np.ndarray[DTYPE_t, ndim=1] coeffs,
                  np.ndarray[DTYPE_t, ndim=1] ff,
                  double bg):
    cdef int npix = xpix.shape[0], j
    cdef double tmp, f
    cdef np.ndarray[DTYPE_t, ndim=1] flux = np.zeros_like(xpix, dtype=DTYPE)
    for i in range(npix):
        f = 0.0
        # for j in range(NUM_INT_TIME):
            # evaluate_psf[double](<double*>max_fracs.data, xpix[i], ypix[i],
            #                      &(<double>coords[2*j]), &(<double>coords[2*j+1]),
            #                      <double*>coeffs.data, &tmp)
            # f += model * tmp / NUM_INT_TIME
        flux[i] = (bg + f) * ff[i]
    return flux
