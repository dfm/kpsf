#ifndef _KPSF_PSF_H_
#define _KPSF_PSF_H_

#include <cmath>

namespace kpsf {

//
// Evaluate a single Gaussian.
//
template <typename T>
bool gaussian_eval (const T& dx, const T& dy, const T* params, T* value)
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


//
// Evaluate the PSF model at a given location away from the center.
//
template <typename T>
bool evaluate_psf (const unsigned int n_psf_comp, const T* params,
                   const T& dx, const T& dy, T* value)
{
    unsigned i, ind;
    T total_weight = T(0.0), w, xoff, yoff,
      tmp, val = T(0.0);

    // Evaluate the first Gaussian. Centered by definition.
    bool flag = gaussian_eval (dx, dy, params, value);
    if (!flag) return false;

    // Evaluate the other Gaussians.
    for (i = 0; i < n_psf_comp - 1; ++i) {
        ind = 3 + i * 6;

        // Compute the amplitude of this component.
        w = 1.0 / (1.0 + exp(-params[ind]));
        total_weight += w;
        if (total_weight > 1.0) return false;

        // Compute the offset between the center of the PSF and this component.
        xoff = params[ind+1];
        yoff = params[ind+2];
        flag = gaussian_eval (dx-xoff, dy-yoff, &(params[ind+3]), &tmp);
        if (!flag) return false;
        val += w * tmp;
    }

    // Add the primary Gaussian back into the other components.
    *value = (*value) * (1.0 - total_weight) + val;
    return true;
}


//
// Compute the pixel value at a given position for a set of parameters.
//
template <typename T>
bool evaluate_pixel (
    const double x0,            // The coordinates of the pixel.
    const double y0,            // ...
    const unsigned n_time,      // The number of time steps to integrate.
    const unsigned n_stars,     // The number of stars in the field.
    const unsigned n_psf_comp,  // The number of PSF components.
    const T* fluxes,            // The n_stars fluxes.
    const T* origin,            // The 2-vector coords of the frame.
    const T* offsets,           // The (n_stars,2) offset vectors for each star.
    const T* psfpars,           // The PSF parameters.
    const T* bkg,               // The background level.
    const T* response,          // The response in the pixel.
    T* value                    // The computed flux. (output)
)
{
    unsigned t, i;
    T dx, dy, val;
    value[0] = T(0.0);
    for (t = 0; t < n_time; ++t) {
        for (i = 0; i < n_stars; ++i) {
            // Compute the coordinates of the star relative to the pixel.
            dx = origin[2*t  ] + offsets[2*i  ] - x0;
            dy = origin[2*t+1] + offsets[2*i+1] - y0;
            if (!(evaluate_psf (n_psf_comp, psfpars, dx, dy, &val)))
                return false;
            value[0] += fluxes[i] * val / T(n_time);
        }
    }
    value[0] = response[0] * (value[0] + bkg[0]);
    return true;
}

};

#endif
