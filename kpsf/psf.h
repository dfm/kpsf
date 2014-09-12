#ifndef _KPSF_PSF_H_
#define _KPSF_PSF_H_

#include <cmath>

namespace kpsf {

template <typename T>
bool evaluate_gaussian_psf (const double x, const double y,
                            const T* coords, const T* params,
                            T* value)
{
    T f0 = coords[0],
      x0 = coords[1],
      y0 = coords[2],
      xvar = params[0],
      yvar = params[1],
      covar = params[2],
      det = xvar * yvar - covar * covar;
    if (det <= 0.0) return false;
    T dx = x0 - x,
      dy = y0 - y,
      arg = (dx * (yvar*dx - covar*dy) + dy * (xvar*dy - covar*dx)) / det;
    *value = f0 * exp(-0.5 * arg) / (2 * M_PI * sqrt(det));
    return true;
}

};

#endif
