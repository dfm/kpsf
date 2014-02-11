#ifndef _KPSF_H_
#define _KPSF_H_

#include "psf.h"

using kpsf::MixtureBasis;

namespace kpsf {

int photometry (MixtureBasis* basis,
                const int nt, const int nx, const int ny,
                const double* flux_imgs, const double* ferr_imgs,
                double* coeffs, double* coords, double* ff, double* bg);

};

#endif
