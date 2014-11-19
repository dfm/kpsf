#ifndef _KPSF_RESIDUAL_H_
#define _KPSF_RESIDUAL_H_

#include "model.h"

namespace kpsf {

class PixelResidual {
public:
    PixelResidual (
        const double x0,            // The coordinates of the pixel.
        const double y0,            // ...
        const unsigned n_time,      // The number of time steps to integrate.
        const unsigned n_stars,     // The number of stars in the field.
        const unsigned n_psf_comp,  // The number of PSF components.
        const double flux,          // The measured flux in this pixel.
        const double flux_istd      // The inverse uncertainty on flux.
    ) : n_time_(n_time), n_stars_(n_stars), n_psf_comp_(n_psf_comp),
        x0_(x0), y0_(y0), flux_(flux), flux_istd_(flux_istd) {};

    template <typename T>
    bool operator() (
            const T* fluxes,     // The n_stars fluxes.
            const T* origin,     // The 2-vector coords of the frame.
            const T* offsets,    // The (n_stars,2) offset vectors for each star.
            const T* psfpars,    // The PSF parameters.
            const T* bkg,        // The background level.
            const T* response,   // The response in the pixel.
            T* resid) const
    {
        T value;
        if (!(evaluate_pixel (x0_, y0_, n_time_, n_stars_, n_psf_comp_,
                              fluxes, origin, offsets, psfpars, bkg, response,
                              &value)))
            return false;
        resid[0] = (value - flux_) * flux_istd_;
        return true;
    };

private:
    unsigned n_time_, n_stars_, n_psf_comp_;
    double x0_, y0_, flux_, flux_istd_;
};

class L2Regularization {
public:
    L2Regularization (const double mean, const double std)
        : mean_(mean), istd_(1.0/std) {};

    template <typename T>
    bool operator() (const T* value, T* residuals) const {
        residuals[0] = (value[0] - T(mean_)) * T(istd_);
        return true;
    };

private:
    double mean_, istd_;
};

// class PSFPrior {
// public:
//     PSFPrior (const double pos_std, const double det_std)
//         : pos_strength_(1.0 / pos_std), det_strength_(1.0 / det_std) {};
//
//     template <typename T>
//     bool operator() (const T* params, T* residuals) const {
//         T xoff, yoff, r2,
//           xvar = params[0],
//           yvar = params[1],
//           covar = params[2],
//           det0 = xvar * yvar - covar * covar, det1;
//         for (int i = 0; i < n_psf_comp_ - 1; ++i) {
//             int ind = 3 + i * 6;
//
//             // Compute the regularization on the position.
//             xoff = params[ind+1];
//             yoff = params[ind+2];
//             r2 = xoff*xoff + yoff*yoff;
//             if (r2 > 0.0) residuals[2*i] = pos_strength_ * sqrt(r2);
//             else residuals[2*i] = T(0.0);
//
//             // Compute the regularization on the relative determinant.
//             xvar = params[ind+3];
//             yvar = params[ind+4];
//             covar = params[ind+5];
//             det1 = xvar * yvar - covar * covar;
//             if (det1 < det0) residuals[2*i+1] = det_strength_ * (det0 - det1);
//             else residuals[2*i+1] = T(0.0);
//             det0 = det1;
//         }
//         return true;
//     };
//
// private:
//     double pos_strength_, det_strength_;
// };

class CentroidRegularization {
public:
    CentroidRegularization (const unsigned n_int, const double std)
        : n_int_(n_int), strength_(1.0 / std) {};

    template <typename T>
    bool operator() (const T* x, T* residuals) const {
        unsigned i;
        for (i = 0; i < n_int_; ++i) {
            residuals[2*i  ] = strength_ * x[2*i  ];
            residuals[2*i+1] = strength_ * x[2*i+1];
        }
        return true;
    };

private:
    unsigned n_int_;
    double strength_;
};

};

#endif
