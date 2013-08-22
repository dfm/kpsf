#ifndef _PIXEL_RESIDUAL_H_
#define _PIXEL_RESIDUAL_H_

#include <cmath>

//
// Evaluate a single-Gaussian PSF.
//
template <typename T>
T evaluate_psf (const T d1, const T d2, const T* const pars)
{
    T invdet = T(0.5) / (pars[0] * pars[2] - pars[1] * pars[1]),
      factor = sqrt(invdet) / T(M_PI);
    return factor * exp(-invdet * (pars[2] * d1 * d1 +
                                    pars[0] * d2 * d2 -
                                    T(2) * pars[1] * d1 * d2));
}

class PixelResidual {
public:
    PixelResidual (double *dim, double *data)
        : dim_(dim), data_(data) {};

    template <typename T>
    bool operator() (const T* const coords, const T* const psfpars,
                     T* residual) const {
        // Check to make sure that the point is within the image.
        if (coords[0] < T(0) || coords[0] > T(dim_[0]) ||
            coords[1] < T(0) || coords[1] > T(dim_[1])) return false;

        // Evaluate the PSF at the pixel location.
        T value = coords[2] * evaluate_psf (coords[0] - T(data_[0]),
                                            coords[1] - T(data_[1]),
                                            psfpars);

        // Compute the residual.
        residual[0] = (T(data_[2]) - value) / T(data_[3]);

        return true;
    }

private:
    double *dim_, *data_;
};

#endif
