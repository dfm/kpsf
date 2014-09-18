#ifndef _KPSF_RESIDUAL_H_
#define _KPSF_RESIDUAL_H_

#include "psf.h"

namespace kpsf {

class PixelResidual {
public:
    PixelResidual (const double pixel_x, const double pixel_y,
                   const double flux, const double flux_istd)
        : pixel_x_(pixel_x), pixel_y_(pixel_y), flux_(flux),
          flux_istd_(flux_istd) {};

    template <typename T>
    bool operator() (const T* coords, const T* psfpars,
                     const T* background, const T* response,
                     T* residuals) const {
        return compute(coords, psfpars, background, response, residuals);
    };

    template <typename T>
    bool compute (const T* coords, const T* psfpars,
                  const T* background, const T* response,
                  T* residuals) const {
        T value;
        if (!(evaluate_dbl_gaussian_psf<T>(0.2, pixel_x_, pixel_y_, coords,
                                           psfpars, &value))) return false;
        value = response[0] * value + background[0];
        residuals[0] = (value - flux_) * flux_istd_;
        return true;
    };

private:
    double pixel_x_, pixel_y_, flux_, flux_istd_;
};

class CalibratedPixelResidual : public PixelResidual {
public:
    CalibratedPixelResidual (const double pixel_x, const double pixel_y,
                             const double flux, const double flux_istd)
        : PixelResidual(pixel_x, pixel_y, flux, flux_istd) {};

    template <typename T>
    bool operator() (const T* coords, const T* psfpars,
                     const T* background, T* residuals) const {
        T response = T(1.0);
        return compute(coords, psfpars, background, &response, residuals);
    };
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

};

#endif
