# distutils: language = c++
from __future__ import division

cimport cython

import numpy as np
cimport numpy as np
np.import_array()

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef extern from "constants.h":
    cdef int NUM_INT_TIME
N_INT_TIME = NUM_INT_TIME

cdef extern from "model.h" namespace "kpsf":
    cdef int evaluate_pixel[T] (
        const double x0,            # The coordinates of the pixel.
        const double y0,            # ...
        const unsigned n_time,      # The number of time steps to integrate.
        const unsigned n_stars,     # The number of stars in the field.
        const unsigned n_psf_comp,  # The number of PSF components.
        const T* fluxes,            # The n_stars fluxes.
        const T* frame_center,      # The 2-vector coords of the frame.
        const T* offsets,           # The (n_stars,2) offset vectors for each star.
        const T* psfpars,           # The PSF parameters.
        const T* bkg,               # The background level.
        const T* response,          # The response in the pixel.
        T* value                    # The computed flux. (output)
    )

    cdef int evaluate_psf[T] (
        const unsigned n_psf_comp,
        const T* params,
        const T& dx,
        const T& dy,
        T* value
    )


def compute_psf (np.ndarray[DTYPE_t, ndim=1, mode="c"] psfpars,
                 double x, double y):
    cdef unsigned n_psf_comp = (psfpars.shape[0] + 3) // 6
    if (psfpars.shape[0] + 3) % 6 != 0:
        raise ValueError("Invalid number of PSF components")
    cdef double value
    cdef int flag = evaluate_psf[double] (n_psf_comp, <double*>psfpars.data,
                                          x, y, &value)
    if not flag:
        raise RuntimeError("Failed to evaluate the PSF")

    return value


def compute_scene (np.ndarray[DTYPE_t, ndim=1, mode="c"] xpix,
                   np.ndarray[DTYPE_t, ndim=1, mode="c"] ypix,
                   np.ndarray[DTYPE_t, ndim=1, mode="c"] fluxes,
                   np.ndarray[DTYPE_t, ndim=2, mode="c"] origin,
                   np.ndarray[DTYPE_t, ndim=2, mode="c"] offsets,
                   np.ndarray[DTYPE_t, ndim=1, mode="c"] psfpars,
                   double bkg,
                   np.ndarray[DTYPE_t, ndim=1, mode="c"] response):
    cdef unsigned npix = xpix.shape[0]
    if npix != ypix.shape[0] or npix != response.shape[0]:
        raise ValueError("Pixel number mismatch")

    cdef unsigned n_stars = fluxes.shape[0]
    if n_stars != offsets.shape[0] or offsets.shape[1] != 2:
        raise ValueError("Offset matrix shape mismatch")

    cdef unsigned n_time = origin.shape[0]
    if origin.shape[1] != 2:
        raise ValueError("Frame center shape mismatch")

    cdef unsigned n_psf_comp = (psfpars.shape[0] + 3) // 6
    if (psfpars.shape[0] + 3) % 6 != 0:
        raise ValueError("Invalid number of PSF components")

    cdef int i, flag
    cdef double val
    cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] result = \
            np.empty(npix, dtype=DTYPE)
    for i in range(npix):
        flag = evaluate_pixel[double](
                            xpix[i], ypix[i], n_time, n_stars, n_psf_comp,
                            <double*>fluxes.data, <double*>origin.data,
                            <double*>offsets.data, <double*>psfpars.data,
                            &bkg, &(response[i]), &val)
        result[i] = val
        if not flag:
            raise RuntimeError("Failed to evaluate the model")

    return result


cdef extern from "kpsf.h" namespace "kpsf":

    cdef cppclass Solver:
        Solver (
            const unsigned nt,
            const unsigned nx,
            const unsigned ny,
            const unsigned n_int,
            const unsigned n_star,
            const unsigned n_psf_comp,
            double* fluxes,     # The n_stars fluxes.
            double* origin,     # The 2-vector coords of the frame.
            double* offsets,    # The (n_stars,2) offset vectors for each star.
            double* psfpars,    # The PSF parameters.
            double* bkg,        # The background level.
            double* response    # The response in the pixel.
        )

        int add_data_point (
            const unsigned t,
            const unsigned xi,
            const unsigned yi,
            const double flux,
            const double ferr)

        void run ()


def solve (time_series,
           np.ndarray[DTYPE_t, ndim=2, mode="c"] fluxes,
           np.ndarray[DTYPE_t, ndim=3, mode="c"] origin,
           np.ndarray[DTYPE_t, ndim=2, mode="c"] offsets,
           np.ndarray[DTYPE_t, ndim=1, mode="c"] psfpars,
           np.ndarray[DTYPE_t, ndim=1, mode="c"] background,
           np.ndarray[DTYPE_t, ndim=2, mode="c"] response):
    # Parse the dimensions.
    cdef unsigned nt = fluxes.shape[0]
    cdef unsigned ns = fluxes.shape[1]
    cdef unsigned n_int = origin.shape[1]
    cdef unsigned nx = response.shape[0]
    cdef unsigned ny = response.shape[1]

    if (origin.shape[0] != nt or origin.shape[2] != 2 or
            offsets.shape[0] != ns or offsets.shape[1] != 2 or
            background.shape[0] != nt or
            time_series.shape[0] != nx or time_series.shape[1] != ny):
        raise ValueError("Incompatible dimensions")

    cdef unsigned n_psf_comp = (psfpars.shape[0] + 3) // 6
    if (psfpars.shape[0] + 3) % 6 != 0:
        raise ValueError("Invalid number of PSF components")

    # Initialize the solver.
    cdef Solver* solver = new Solver(nt, nx, ny, n_int, ns, n_psf_comp,
                                     <double*>(fluxes.data),
                                     <double*>(origin.data),
                                     <double*>(offsets.data),
                                     <double*>(psfpars.data),
                                     <double*>(background.data),
                                     <double*>(response.data))

    # Loop over the frames and add the pixels to the model.
    cdef unsigned t, xi, yi
    cdef int flag
    for t in range(nt):
        if not time_series.good_times[t]:
            continue
        frame = time_series.frames[t]
        for xi in range(time_series.shape[0]):
            for yi in range(time_series.shape[1]):
                if not np.isfinite(frame.img[xi, yi]):
                    continue
                flag = solver.add_data_point(t, xi, yi,
                                             frame.img[xi, yi],
                                             frame.err_img[xi, yi])
                if flag != 0:
                    raise RuntimeError("Couldn't add data point")

    # Run the solver.
    solver.run()
    del solver
