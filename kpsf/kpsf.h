#ifndef _KPSF_KPSF_H_
#define _KPSF_KPSF_H_

namespace kpsf {

int photometry_one (const int npix, const double* xpix, const double* ypix,
                    const double* flux, const double* ferr,
                    double* coords, double* coeffs, double* bg);

int photometry_all (const int nt, const int npix, const double* xpix,
                    const double* ypix, const double* flux,
                    const double* ferr, double* coords, double* coeffs,
                    double* ff, double* bg);

};

#endif