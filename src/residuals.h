#ifndef _RESIDUALS_H_
#define _RESIDUALS_H_

#include <cmath>

namespace kpsf {

template <typename PSFType>
class PixelResidual {

public:

    PixelResidual (PSFType* psf, const int i, const int j,
                   const int minx, const int maxx, const int miny,
                   const int maxy, const double mean, const double std)
        : i_(i), j_(j), minx_(minx), maxx_(maxx), miny_(miny), maxy_(maxy),
          mean_(mean), istd_(1.0/std), psf_(psf) {};

    template <typename T>
    bool operator() (const T* coords, const T* psfpars,
                     const T* background, const T* response,
                     T* residuals) const {
        // Compute the pixel position in PRF coordinates.
        T value;

        // Check the boundaries.
        if (coords[1] < T(minx_) || coords[1] > T(maxx_) ||
            coords[2] < T(miny_) || coords[2] > T(maxy_)) return false;

        // Interpolate the PSF to the position of the pixel.
        if (! (predict(coords, psfpars, background, response, &value)))
            return false;

        // Compute the residuals.
        residuals[0] = (value - T(mean_)) * T(istd_);
        return true;
    };

    template <typename T>
    bool predict (const T* coords, const T* psfpars, const T* background,
                  const T* response, T* value) const {
        // Compute the relative pixel position.
        T xi = T(i_) - coords[1],
          yi = T(j_) - coords[2];

        // Evaluate the PSF model.
        if (! (psf_->evaluate<T>(psfpars, xi, yi, value))) return false;

        // Apply the response and background.
        *value = response[0] * (coords[0] * value[0] + background[0]);
        return true;
    };

private:

    PSFType* psf_;
    double i_, j_, minx_, maxx_, miny_, maxy_, mean_, istd_;

};

class SumToOneResidual {

public:

    SumToOneResidual (int N, double strength)
        : number_(N), strength_(strength) {};

    template <typename T>
    bool operator() (const T* coeffs, T* residuals) const {
        residuals[0] = T(1.0);
        for (int i = 0; i < number_; ++i)
            residuals[0] -= coeffs[i];
        residuals[0] *= T(strength_);
        return true;
    };

private:

    int number_;
    double strength_;

};

class L2Residual {

public:

    L2Residual (int N, double mean, double strength)
        : number_(N), mean_(mean), strength_(strength) {};

    template <typename T>
    bool operator() (const T* coeffs, T* residuals) const {
        for (int i = 0; i < number_; ++i)
            residuals[i] = T(strength_) * (coeffs[i] - T(mean_));
        return true;
    };

private:

    int number_;
    double mean_, strength_;

};

};

#endif
