from __future__ import division

cimport cython

import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef extern from "kpsf.h" namespace "kpsf":
    cdef int photometry_all (const int nt, const int npix,
                             const double maxx, const double maxy,
                             const double* xpix,
                             const double* ypix, const double* flux,
                             const double* ferr, double* model,
                             double* coords, double* coeffs, double* ff,
                             double* bg, const double* max_fracs,
                             const double motion_reg, const double ff_reg)

cdef extern from "residual.h":
    cdef int NUM_INT_TIME
N_INT_TIME = NUM_INT_TIME

cdef extern from "psf.h":
    cdef int NUM_PSF_COMP
N_PSF_COMP = NUM_PSF_COMP

cdef extern from "psf.h" namespace "kpsf":
    cdef int evaluate_psf[T] (const double* max_fracs,
                              const double x, const double y,
                              const T* coords, const T* params,
                              T* value)


def run_photometry_all(np.ndarray[DTYPE_t, ndim=1] time,
                       np.ndarray[DTYPE_t, ndim=1] xpix,
                       np.ndarray[DTYPE_t, ndim=1] ypix,
                       np.ndarray[DTYPE_t, ndim=2] flux,
                       np.ndarray[DTYPE_t, ndim=2] ferr,
                       np.ndarray[DTYPE_t, ndim=1] model,
                       np.ndarray[DTYPE_t, ndim=2] coords,
                       np.ndarray[DTYPE_t, ndim=1] coeffs,
                       np.ndarray[DTYPE_t, ndim=1] ff,
                       np.ndarray[DTYPE_t, ndim=1] bg,
                       np.ndarray[DTYPE_t, ndim=1] max_fracs,
                       double motion_reg, double ff_reg):
    cdef int nt = time.shape[0]
    cdef int npix = xpix.shape[0]
    if (npix != ypix.shape[0] or nt != flux.shape[0] or npix != flux.shape[1]
            or nt != ferr.shape[0] or npix != ferr.shape[1]
            or model.shape[0] != nt
            or coords.shape[0] != nt
            or coords.shape[1] != 2 * NUM_INT_TIME
            or coeffs.shape[0] != 6*NUM_PSF_COMP-3 or bg.shape[0] != nt
            or ff.shape[0] != npix or max_fracs.shape[0] != NUM_PSF_COMP-1):
        raise ValueError("Invalid dimensions")

    photometry_all(nt, npix, <double>(xpix.max()+1), <double>(ypix.max()+1),
                   <double*>xpix.data, <double*>ypix.data,
                   <double*>flux.data, <double*>ferr.data,
                   <double*>model.data,
                   <double*>coords.data, <double*>coeffs.data,
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
        for j in range(NUM_INT_TIME):
            evaluate_psf[double](<double*>max_fracs.data, xpix[i], ypix[i],
                                 <double*>coords.data + 2 * j,
                                 <double*>coeffs.data, &tmp)
            f += model * tmp / NUM_INT_TIME
        flux[i] = (bg + f) * ff[i]
    return flux
