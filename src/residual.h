#ifndef _RESIDUAL_H_
#define _RESIDUAL_H_

#include <iostream>
#include <cmath>

#define NGAUSSIANS 1
#define NPSFPARS   4 * NGAUSSIANS - 1

class PixelResidual {
public:
    PixelResidual (double *data)
        : data_(data) {};

    template <typename T>
    bool operator() (const T* const coords, const T* const psfpars,
                     const T* const ff, T* residuals) const {
        int k, ind;
        T value, norm, det, invdet, factor, d1, d2;

        // Compute the pixel coordinate offsets.
        d1 = coords[0] - T(data_[0]);
        d2 = coords[1] - T(data_[1]);

        // Loop over the Gaussians and compute the PSF value.
        norm = T(1.0);
        value = T(0.0);
        for (k = 0; k < NGAUSSIANS; ++k) {
            ind = 4 * k - 1;
            det = psfpars[ind+1]*psfpars[ind+3]-psfpars[ind+2]*psfpars[ind+2];
            if (det <= T(0.0)) return false;
            invdet = T(0.5) / det;
            factor = sqrt(invdet);
            if (ind > 0) {
                if (psfpars[ind] < 0.0 || psfpars[ind] > 1.0) return false;
                factor *= psfpars[ind];
                norm += psfpars[ind];
            }
            value += factor * exp(-invdet *
                                  (psfpars[ind+3] * d1 * d1
                                   + psfpars[ind+1] * d2 * d2
                                   - T(2) * psfpars[ind+2] * d1 * d2));
        }

        value *= ff[0] * coords[2] / norm / T(M_PI);
        residuals[0] = (T(data_[2]) - value) * T(data_[3]);
        return true;
    };

private:
    double *data_;
};

#endif
