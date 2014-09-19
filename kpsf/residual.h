#ifndef _KPSF_RESIDUAL_H_
#define _KPSF_RESIDUAL_H_

#include "psf.h"

#define NUM_INT_TIME 3

namespace kpsf {

class PixelResidual {
public:
    PixelResidual (const double maxx, const double maxy,
                   const double pixel_x, const double pixel_y,
                   const double flux, const double flux_istd,
                   const double* max_fracs)
        : maxx_(maxx), maxy_(maxy), pixel_x_(pixel_x), pixel_y_(pixel_y),
          flux_(flux), flux_istd_(flux_istd), max_fracs_(max_fracs) {};

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
            if (pixel_x_ < 0 || pixel_x_ > maxx_ ||
                    pixel_y_ < 0 || pixel_y_ > maxy_) return false;
            if (!(evaluate_psf<T>(max_fracs_, pixel_x_, pixel_y_,
                                  &(coords[2*i]), psfpars, &tmp)))
                return false;
            value += tmp / T(NUM_INT_TIME);
        }
        value = response[0] * (flux[0] * value + background[0]);
        residuals[0] = (value - flux_) * flux_istd_;
        return true;
    };

private:
    double maxx_, maxy_, pixel_x_, pixel_y_, flux_, flux_istd_;
    const double* max_fracs_;
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
    bool operator() (const T* x, T* residuals) const {
        T xoff, yoff, d;
        for (int i = 0; i < NUM_INT_TIME - 1; ++i) {
            xoff = x[2*i+2] - x[2*i];
            yoff = x[2*i+3] - x[2*i+1];
            d = xoff * xoff + yoff * yoff;
            residuals[i] = strength_ * sqrt(xoff * xoff + yoff * yoff);
        }
        return true;
    };

private:
    double strength_;
};

};

#endif
