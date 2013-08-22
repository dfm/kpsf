#ifndef _KPSF_SOLVER_H_
#define _KPSF_SOLVER_H_

#ifdef __cplusplus
extern "C"
#endif
int kpsf_solve (int ntime, int npixels, double *data, double *dim,
                double *coords, double *psfpars, int verbose);

#endif
