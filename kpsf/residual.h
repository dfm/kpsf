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
        if (!(evaluate_gaussian_psf<T>(pixel_x_, pixel_y_, coords, psfpars,
                                       &value))) return false;
        value = response[0] * (value + background[0]);
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

};

#endif
