#ifndef _KPSF_RESIDUAL_H_
#define _KPSF_RESIDUAL_H_

#include "psf.h"
#include "constants.h"

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
        T value = T(0.0), tmp, x, y;
        for (int i = 0; i < NUM_INT_TIME; ++i) {
            x = coords[2*i];
            y = coords[2*i+1];
            if (x < 0.0 || x > maxx_ || y < 0.0 || y > maxy_) return false;
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
    PSFPrior (const double pos_std, const double det_std)
        : pos_strength_(1.0 / pos_std), det_strength_(1.0 / det_std) {};

    template <typename T>
    bool operator() (const T* params, T* residuals) const {
        T xoff, yoff, r2,
          xvar = params[0],
          yvar = params[1],
          covar = params[2],
          det0 = xvar * yvar - covar * covar, det1;
        for (int i = 0; i < NUM_PSF_COMP - 1; ++i) {
            int ind = 3 + i * 6;

            // Compute the regularization on the position.
            xoff = params[ind+1];
            yoff = params[ind+2];
            r2 = xoff*xoff + yoff*yoff;
            if (r2 > 0.0) residuals[2*i] = pos_strength_ * sqrt(r2);
            else residuals[2*i] = T(0.0);

            // Compute the regularization on the relative determinant.
            xvar = params[ind+3];
            yvar = params[ind+4];
            covar = params[ind+5];
            det1 = xvar * yvar - covar * covar;
            if (det1 < det0) residuals[2*i+1] = det_strength_ * (det0 - det1);
            else residuals[2*i+1] = T(0.0);
            det0 = det1;
        }
        return true;
    };

private:
    double pos_strength_, det_strength_;
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
