#ifndef _KPSF_PSF_H_
#define _KPSF_PSF_H_

#include <cmath>
#include "constants.h"

namespace kpsf {

template <typename T>
bool gaussian_eval (const T dx, const T dy, const T* params, T* value)
{
    T xvar = params[0],
      yvar = params[1],
      covar = params[2],
      det = xvar * yvar - covar * covar;
    if (det <= 0.0) return false;
    T arg = (dx * (yvar*dx - covar*dy) + dy * (xvar*dy - covar*dx)) / det;
    *value = exp(-0.5 * arg) / (2 * M_PI * sqrt(det));
    return true;
}

template <typename T>
bool evaluate_psf (const double* max_fracs, const T* params,
                   const double x, const double y,
                   const T& x0, const T& y0, T* value)
{
    T total_weight = T(0.0), w, xoff, yoff,
      tmp, val = T(0.0);

    // Evaluate the first Gaussian.
    bool flag = gaussian_eval (x - x0, y - y0, params, value);
    if (!flag) return false;

    // Evaluate the other Gaussians.
    for (int i = 0; i < NUM_PSF_COMP - 1; ++i) {
        int ind = 3 + i * 6;
        w = max_fracs[i] / (1.0 + exp(-params[ind]));
        total_weight += w;
        if (total_weight > 1.0) return false;
        xoff = params[ind+1];
        yoff = params[ind+2];
        flag = gaussian_eval (x-x0-xoff, y-y0-yoff, &(params[ind+3]), &tmp);
        if (!flag) return false;
        val += w * tmp;
    }

    *value = (*value) * (1.0 - total_weight) + val;
    return true;
}

};

#endif
