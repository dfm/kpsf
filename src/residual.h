#ifndef _PIXEL_RESIDUAL_H_
#define _PIXEL_RESIDUAL_H_

#include <iostream>
#include <cmath>

template <typename T>
class PSF {
public:
    PSF (const T* const pars)
        : pars_(pars)
    {
        det_ = pars_[0] * pars_[2] - pars_[1] * pars_[1];
        invdet_ = T(0.5) / det_;
        factor_ = sqrt(invdet_) / T(M_PI);
    };

    T evaluate (const T d1, const T d2) const {
        return factor_ * exp(-invdet_ * (pars_[2] * d1 * d1 +
                                         pars_[0] * d2 * d2 -
                                         T(2) * pars_[1] * d1 * d2));
    };

    T get_det () { return det_; };

private:
    const T* const pars_;
    T det_, invdet_, factor_;
};

class KPSFResidual {
public:
    KPSFResidual (int ntime, int npixels, double *data, double *dim)
        : ntime_(ntime), npixels_(npixels), dim_(dim), data_(data) {};

    template <typename T>
    bool operator() (T const* const* pars, T* residuals) const {
        int i, j, ind, ind4, i3;
        T value;
        PSF<T> psf(pars[2]);
        if (psf.get_det() <= T(0.0)) return false;

        for (i = 0; i < ntime_; ++i) {
            i3 = 3*i;
            if (pars[0][i3] < T(0.0) || pars[0][i3] > T(dim_[0]) ||
                pars[0][i3+1] < T(0.0) || pars[0][i3+1] > T(dim_[1]) ||
                pars[0][i3+2] < T(0.0)) return false;
            for (j = 0; j < npixels_; ++j) {
                ind = i*npixels_+j;
                ind4 = 4*ind;
                value = pars[1][j] * pars[0][i3+2] *
                        psf.evaluate(pars[0][i3]-T(data_[ind4]),
                                     pars[0][i3+1]-T(data_[ind4+1]));
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
