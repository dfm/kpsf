#ifndef _KPSF_SOLVER_H_
#define _KPSF_SOLVER_H_


//
// Solve for the flux time series.
//
// :param int ntime:
//      The number of time samples.
//
// :param int npixels:
//      The number of unmasked pixels.
//
// :param double ***data:
//      The data. This object should have the shape:
//          (ntime, npixels, 4)
//      and the last dimension should be (xpos, ypos, flux, inv_ferr)
//
// :param double *dim:
//      The dimensions of the image.
//
// :param double **coords:
//      An initial guess for the coordinate time series of the source
//      (xpos, ypos, flux).
//
// :param double *psfpars:
//      An initial guess for the PSF parameters. Must have length 3.
//

#ifdef __cplusplus
extern "C"
#endif
int kpsf_solve (int ntime, int npixels, double *data, double *dim,
                double *coords, double *psfpars);

#endif
