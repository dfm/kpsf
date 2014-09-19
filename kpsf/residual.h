#ifndef _KPSF_RESIDUAL_H_
#define _KPSF_RESIDUAL_H_

#include "psf.h"

#define NUM_INT_TIME 5

namespace kpsf {

class PixelResidual {
public:
    PixelResidual (const double pixel_x, const double pixel_y,
                   const double flux, const double flux_istd,
                   const double max_frac)
        : pixel_x_(pixel_x), pixel_y_(pixel_y), flux_(flux),
          flux_istd_(flux_istd), max_frac_(max_frac) {};

    template <typename T>
    bool operator() (const T* flux, const T* coords,
                     const T* psfpars,
                     const T* background, const T* response,
                     T* residuals) const {
        return compute(flux, coords, psfpars, background, response, residuals);
    };

    template <typename T>
    bool compute (const T* flux, const T* coords, const T* psfpars,
                  const T* background, const T* response,
                  T* residuals) const {
        T value = T(0.0), tmp;
        for (int i = 0; i < NUM_INT_TIME; ++i) {
            if (!(evaluate_dbl_gaussian_psf<T>(max_frac_, pixel_x_, pixel_y_,
                                               &(coords[2*i]), psfpars, &tmp)))
                return false;
            value += tmp / T(NUM_INT_TIME);
        }
        value = response[0] * (flux[0] * value + background[0]);
        residuals[0] = (value - flux_) * flux_istd_;
        return true;
    };

private:
    double pixel_x_, pixel_y_, flux_, flux_istd_, max_frac_;
};

class GaussianPrior {
public:
    GaussianPrior (const double mean, const double std)
        : mean_(mean), istd_(1.0/std) {};

    template <typename T>
    bool operator() (const T* value, T* residuals) const {
        residuals[0] = (value[0] - T(mean_)) * T(istd_);
        return true;
    };

private:
    double mean_, istd_;
};

class PSFPrior {
public:
    PSFPrior (const double std)
        : strength_(1.0 / std) {};

    template <typename T>
    bool operator() (const T* psfpars, T* residuals) const {
        T xoff = psfpars[0],
          yoff = psfpars[1];
        residuals[0] = strength_ * sqrt(xoff * xoff + yoff * yoff);
        return true;
    };

private:
    double strength_;
};

class SmoothPrior {
public:
    SmoothPrior (const double std)
        : strength_(1.0 / std) {};

    template <typename T>
    bool operator() (const T* x1, const T* x2, T* residuals) const {
        T xoff, yoff;
        for (int i = 0; i < NUM_INT_TIME - 1; ++i) {
            xoff = x1[2*i+2] - x1[2*i];
            yoff = x1[2*i+3] - x1[2*i+1];
            residuals[i] = strength_ * sqrt(xoff * xoff + yoff * yoff);
        }
        xoff = x2[0] - x1[2*NUM_INT_TIME-2];
        xoff = x2[1] - x1[2*NUM_INT_TIME-1];
        residuals[NUM_INT_TIME-1] = strength_ * sqrt(xoff * xoff + yoff * yoff);
        return true;
    };

private:
    double strength_;
};

};

#endif
