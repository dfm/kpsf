#ifndef _KPSF_PSF_H_
#define _KPSF_PSF_H_

#include <cmath>

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
bool evaluate_gaussian_psf (const double x, const double y,
                            const T* coords, const T* params,
                            T* value)
{
    T f0 = coords[0],
      x0 = coords[1],
      y0 = coords[2];
    bool flag = gaussian_eval (x - x0, y - y0, params, value);
    if (!flag) return false;
    *value *= f0;
    return true;
}

template <typename T>
bool evaluate_dbl_gaussian_psf (const double max_frac,
                                const double x, const double y,
                                const T* coords, const T* params,
                                T* value)
{
    T f0 = coords[0],
      x0 = coords[1],
      y0 = coords[2],
      frac = max_frac / (1.0 + exp(-params[0])),
      val;

    // Evaluate the first Gaussian.
    bool flag = gaussian_eval (x - x0, y - y0, &(params[1]), value);
    if (!flag) return false;

    // Evaluate the second Gaussian.
    T xoff = params[4],
      yoff = params[5];
    flag = gaussian_eval (x - x0 + xoff, y - y0 + yoff, &(params[6]), &val);
    if (!flag) return false;

    *value = f0 * (frac * val + (1.0 - frac) * (*value));
    return true;
}

};

#endif