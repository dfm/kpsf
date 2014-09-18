from __future__ import division

cimport cython

import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef extern from "kpsf.h" namespace "kpsf":

    # cdef int photometry_one (const int npix, const double* xpix,
    #                          const double* ypix, const double* flux,
    #                          const double* ferr,
    #                          double* coords, double* coeffs, double* bg)

    cdef int photometry_all (const int nt, const int npix, const double* xpix,
                             const double* ypix, const double* flux,
                             const double* ferr,
                             double* coords, double* coeffs, double* ff,
                             double* bg)

cdef extern from "psf.h" namespace "kpsf":

    cdef int evaluate_dbl_gaussian_psf[T] (const double max_frac,
                                           const double x, const double y,
                                           const T* coords, const T* params,
                                           T* value)


# def run_photometry_one(np.ndarray[DTYPE_t, ndim=1] xpix,
#                        np.ndarray[DTYPE_t, ndim=1] ypix,
#                        np.ndarray[DTYPE_t, ndim=1] flux,
#                        np.ndarray[DTYPE_t, ndim=1] ferr,
#                        np.ndarray[DTYPE_t, ndim=1] coords,
#                        np.ndarray[DTYPE_t, ndim=1] coeffs,
#                        np.ndarray[DTYPE_t, ndim=1] bg):
#     cdef int npix = xpix.shape[0]
#     if (npix != ypix.shape[0] or npix != flux.shape[0] or npix != ferr.shape[0]
#             or coords.shape[0] != 3 or coeffs.shape[0] != 3
#             or bg.shape[0] != 1):
#         raise ValueError("Invalid dimensions")

#     photometry_one(npix, <double*>xpix.data, <double*>ypix.data,
#                    <double*>flux.data, <double*>ferr.data,
#                    <double*>coords.data, <double*>coeffs.data,
#                    <double*>bg.data)


def run_photometry_all(np.ndarray[DTYPE_t, ndim=1] time,
                       np.ndarray[DTYPE_t, ndim=1] xpix,
                       np.ndarray[DTYPE_t, ndim=1] ypix,
                       np.ndarray[DTYPE_t, ndim=2] flux,
                       np.ndarray[DTYPE_t, ndim=2] ferr,
                       np.ndarray[DTYPE_t, ndim=2] coords,
                       np.ndarray[DTYPE_t, ndim=1] coeffs,
                       np.ndarray[DTYPE_t, ndim=1] ff,
                       np.ndarray[DTYPE_t, ndim=1] bg):
    cdef int nt = time.shape[0]
    cdef int npix = xpix.shape[0]
    if (npix != ypix.shape[0] or nt != flux.shape[0] or npix != flux.shape[1]
            or nt != ferr.shape[0] or npix != ferr.shape[1]
            or coords.shape[0] != nt or coords.shape[1] != 3
            or coeffs.shape[0] != 9 or bg.shape[0] != nt
            or ff.shape[0] != npix):
        raise ValueError("Invalid dimensions")

    photometry_all(nt, npix, <double*>xpix.data, <double*>ypix.data,
                   <double*>flux.data, <double*>ferr.data,
                   <double*>coords.data, <double*>coeffs.data,
                   <double*>ff.data, <double*>bg.data)


def compute_model(np.ndarray[DTYPE_t, ndim=1] xpix,
                  np.ndarray[DTYPE_t, ndim=1] ypix,
                  np.ndarray[DTYPE_t, ndim=1] coords,
                  np.ndarray[DTYPE_t, ndim=1] coeffs,
                  np.ndarray[DTYPE_t, ndim=1] ff,
                  np.ndarray[DTYPE_t, ndim=1] bg):
    cdef int npix = xpix.shape[0]
    cdef double f
    cdef np.ndarray[DTYPE_t, ndim=1] flux = bg[0] + np.zeros_like(xpix,
                                                                  dtype=DTYPE)
    for i in range(npix):
        evaluate_dbl_gaussian_psf[double](0.2, xpix[i], ypix[i],
                                          <double*>coords.data,
                                          <double*>coeffs.data,
                                          &f)
        flux[i] += f * ff[i]
    return flux
