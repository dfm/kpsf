#ifndef _PIXEL_RESIDUAL_H_
#define _PIXEL_RESIDUAL_H_

#include <iostream>
#include <cmath>

class KPSFResidual {
public:
    KPSFResidual (int ntime, int npixels, double *data, double *dim)
        : ntime_(ntime), npixels_(npixels), dim_(dim), data_(data) {};

    template <typename T>
    bool operator() (T const* const* pars, T* residuals) const {
        int i, j, ind, ind4, i3;
        T value, det, invdet, factor, d1, d2;
        det = pars[3][0] * pars[3][2] - pars[3][1] * pars[3][1];
        if (det <= T(0.0)) return false;
        invdet = T(0.5) / det;
        factor = sqrt(invdet) / T(M_PI);

        for (i = 0; i < ntime_; ++i) {
            i3 = 3*i;
            if (pars[0][i3] < T(0.0) || pars[0][i3] > T(dim_[0]) ||
                pars[0][i3+1] < T(0.0) || pars[0][i3+1] > T(dim_[1]) ||
                pars[0][i3+2] < T(0.0)) return false;
            for (j = 0; j < npixels_; ++j) {
                ind = i*npixels_+j;
                ind4 = 4*ind;

                d1 = pars[0][i3]-T(data_[ind4]);
                d2 = pars[0][i3+1]-T(data_[ind4+1]);

                value = pars[2][j] + pars[1][j]*pars[0][i3+2]*factor
                      * exp(-invdet * (pars[3][2]*d1*d1 + pars[3][0]*d2*d2 -
                                       T(2)*pars[3][1]*d1*d2));

                residuals[ind] = (T(data_[ind4+2]) - value) * T(data_[ind4+3]);
            }
        }

        return true;
    };

private:
    int ntime_, npixels_;
    const double * const dim_, * const data_;
};

#endif
