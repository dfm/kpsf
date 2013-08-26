#ifndef _RESIDUAL_H_
#define _RESIDUAL_H_

#include <iostream>
#include <cmath>

class PixelResidual {
public:
    PixelResidual (double *data)
        : data_(data) {};

    template <typename T>
    bool operator() (const T* const coords, const T* const psfpars,
                     const T* const ff, T* residuals) const {
        T value, det, invdet, factor, d1, d2;

        // Compute the PSF.
        det = psfpars[0] * psfpars[2] - psfpars[1] * psfpars[1];
        if (det <= T(0.0)) return false;
        invdet = T(0.5) / det;
        factor = sqrt(invdet) / T(M_PI);

        // Evaluate the PSF at the pixel position.
        d1 = coords[0] - T(data_[0]);
        d2 = coords[1] - T(data_[1]);

        value = ff[0] * coords[2] * factor
                * exp(-invdet * (psfpars[2] * d1 * d1 + psfpars[0] * d2 * d2
                                 - T(2) * psfpars[1] * d1 * d2));

        residuals[0] = (T(data_[2]) - value) * T(data_[3]);
        return true;
    };

private:
    double *data_;
};

#endif
